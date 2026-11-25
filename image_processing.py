"""
image_processing.py — FIR scanned-document image preprocessing (OpenCV + NumPy only, runs locally).

    from image_processing import DocumentPreprocessor, preprocess_pdf

    pre = DocumentPreprocessor()                         # settings + device detected once
    finals = pre.process("doc.pdf")                      # list[np.ndarray] RGB, one per page, page order
                                                         # (ready for matplotlib / PIL / AI models; color="bgr" for OpenCV)
    pages = pre.process("doc.pdf", return_all=True)      # list[PageResult]: .page .original .final .applied
    pre.process("doc.pdf", save=True)                    # also saves to the current directory
    pre.process("doc.pdf", save=True, output_dir="out")  # ... or to out/
    preprocess_pdf("doc.pdf", save=True)                 # one-line shortcut

Saved per page (flat in output_dir, no per-PDF folder):
    {pdf}_{page:03d}_original.jpg   exact scan from the PDF (byte for byte; .png if the page had to be rendered)
    {pdf}_{page:03d}_final.png      cleaned colour page (lossless) - the image to send to the model
    {pdf}_{page:03d}_applied.json   every step in pipeline order: applied?, why, parameters used, result, flags

Pipeline (each step runs only where its own measurement says the page needs it; see README.md):
    1 render (native scan)  2 assess  3 orientation  4 borders  5 deskew  6 illumination
    7 noise  8 sharpen  9 bleed-through  10 faint-ink contrast  11 final image
The step functions below are the reviewed notebook code (app.ipynb imports this module for review).
"""

from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from device import apply_profile, get_device_profile, print_profile

# ====================================================================================================
# Default settings (all sizes in mm, scaled per page by its DPI)
# ====================================================================================================
CONFIG = {
    "input": {
        "pdf_dir": "sample-docs",
        "pdf_files": None,        # None = every PDF in pdf_dir, or a list such as ["FIR-2-2009.pdf"]
        "page_range": None,       # None = all pages, or (first, last) 1-based inclusive, e.g. (1, 3) for quick tests
    },
    "output": {
        "dir": "output",          # results go to output/<doc>/
        "png_compression": 3,     # 0-9 (higher = smaller, slower); PNG is lossless at any level
        "bw": {                   # black & white version (Sauvola adaptive threshold)
            "enabled": False,     # off: the colour final image is what goes to the model; True also makes/saves B&W
            "source": "min_channel",  # "min_channel" (darkest of B,G,R: faint blue ink is much darker in red) or
                                      # "luminance". Measured: min_channel + k 0.15 keeps 98.8% of stroke cores
                                      # (median) vs 92.7% for luminance + k 0.2, with B&W noise still <= 0.15%
            "window_mm": 6.0,     # local window (a few text-line heights)
            "k": 0.15,            # Sauvola k: higher = thinner/lighter strokes
            "R": 128,             # dynamic range of the standard deviation
            "min_blob_mm2": 0.02, # isolated black blobs smaller than this are removed
        },
        "save_stages": None,      # None = follow PROFILE["save_intermediates"]; True/False to force
        "page_digits": 3,         # {doc}_{page}_...: page number zero-padded to at least this many digits
    },
    "batch": {                    # streaming mode for large document sets (see "Batch mode" cell)
        "pages_in_flight": None,  # pages held in memory at once; None = 2 x PROFILE["num_workers"]
        "keep_images": False,     # True keeps every page's images in memory (only for small sets / review)
    },
    "render": {
        "upscale": 1.0,           # >1 enlarges after extraction (small handwriting); capped by PROFILE["max_render_scale"]
        "fallback_dpi": 200,      # only for pages that are not a single full-page scan image
        "assumed_paper_width_in": 8.5,  # PDF says 72 DPI (meaningless); estimate DPI from the short side
        "dpi_override": None,     # set a number if the estimate is wrong (all physical sizes below scale with DPI)
        "dpi_per_document": True, # one scanner session per PDF: use the median page estimate (narrow sheets
                                  # otherwise look low-DPI because the estimate assumes full-width paper)
        "color_scale": 0.25,      # colour stats on a 1/4-size copy
        "color_chroma_min": 40,   # max-min channel difference that counts as a "coloured" pixel
        "color_paper_min": 150,   # pixels brighter than this (all channels) count as paper for the tint
        "greyscale_max_ink_pct": 0.05,  # a scan is "greyscale" if coloured pixels < this % ...
        "greyscale_max_tint": 3,        # ... and paper chroma < this
    },
    "assess": {
        # All sizes are in millimetres and converted per page with its DPI (Step 1),
        # so a 150 / 300 DPI scan gets the same physical behaviour as the 200 DPI samples.
        "work_scale": 4,                # borders/illumination measured on a 1/4-size copy
        "reference_dpi": 200,           # page content is resampled to this DPI before measuring, so pixel-level
                                        # metrics (sharpness, noise, specks) mean the same for any scan resolution
        # --- borders ---
        "paper_ref_percentile": 50,     # paper colour = median of the brightest half of the page centre
        "border_dark_rel": 0.40,        # "dark" = darker than 40% of this page's paper level
        "border_dark_frac": 0.35,       # a line belongs to a dark strip if >= 35% of it is dark
        "border_dark_min_mm": 1.5,      # a dark strip must be at least this wide (table rules are thinner)
        "border_dark_grow_frac": 0.15,  # then grow the strip inward while lines are still >= 15% dark (shadow edge) ...
        "border_dark_grow_max_mm": 3.0, # ... by at most this much
        "border_pale_min_mm": 1.0,      # minimum width of a pale scanner-bed strip
        "border_dark_search": 0.20,     # search for dark strips in the outer 20% of each side
        "border_pale_search": 0.08,     # search for pale strips in the outer 8%
        "border_pale_de": 12,           # pale strip: Lab colour distance from paper >= this ...
        "border_pale_std": 12,          # ... and uniform along the line (lightness std < this; higher = text)
        # --- strokes (black-hat = how much darker than the local background) ---
        "stroke_width_max_mm": 1.9,     # strokes thinner than this count as ink
        "stroke_weak": 12, "stroke_ink": 30, "stroke_strong": 45,  # black-hat grey levels
        "contrast_percentile": 90,      # stroke contrast = this percentile of black-hat on ink
        # --- sharpness / noise ---
        "sharp_sigma_mm": 0.13,         # high-frequency = image - Gaussian(this)
        "sharp_band_mm": 0.38,          # measure on a band this wide around strokes
        "noise_flat_percentile": 90,    # Immerkaer noise on pixels below this gradient percentile
        # --- faint marks (bleed-through vs texture) ---
        "rule_line_min_mm": 7.7,        # straight runs longer than this are ruled/dotted lines
        "rule_line_gap_mm": 1.1,        # gaps bridged when finding dotted lines
        "rule_line_pad_mm": 0.6,        # margin removed around found lines
        "near_ink_mm": 1.1,             # faint marks closer than this to real ink are stroke edges, not bleed
        "faint_stroke_min_area_mm2": 0.65,  # faint component must be at least this big ...
        "faint_stroke_min_size_mm": 1.0,    # ... at least this tall AND wide (curvy, not a line) ...
        "faint_stroke_max_fill": 0.6,       # ... and fill < 60% of its box (stroke, not a blob)
        "paper_pad_mm": 0.6,            # paper pixels = this far from any mark
        # --- specks ---
        "speck_max_area_mm2": 0.10,     # blobs up to this size can be specks
        "speck_isolation_mm": 2.1,      # ... if no other ink in a window this wide
        # --- illumination ---
        "illum_close_mm": 7.6,          # closing removes strokes before estimating the paper level
        "illum_median_mm": 10.7,
        "illum_percentiles": (5, 95),
        # --- ink colours (OpenCV hue 0-180; red wraps around 180) ---
        "ink_chroma_min": 30,
        "ink_hues": {"blue": (90, 130), "purple": (130, 165), "red": (165, 12)},
        # --- what later steps should do ("needs") ---
        "needs": {
            "sharpen_below": 7.0,       # sharpness score -> Step 8
            "denoise_mild_sigma": 1.0,  # noise sigma -> Step 7 (mild)
            "denoise_strong_sigma": 2.0,  # -> Step 7 (strong)
            "denoise_texture_pct": 2.0, # faint dotty texture -> Step 7 (mild)
            "speck_removal_per_mp": 100,  # isolated specks / megapixel -> Step 7
            "bleed_faint_stroke_pct": 1.0,  # faint stroke-like marks -> Step 9
            "illumination_spread": 25,  # paper-level spread -> Step 6
            "paper_tint": 6,            # paper chroma -> Step 6 tint removal
            "faint_ink_level": 180,     # median ink grey level -> Step 10 contrast
        },
        # --- unknown-issue guard: range of each metric seen on the 71 sample pages (with margin).
        # A page outside a range is flagged "unusual_<metric>" for review: it may have a problem we
        # have not designed for. Widen a range only after looking at the page.
        "expected_ranges": {
            "ink_pct": (0.3, 55), "sharpness": (4.5, 20), "noise_sigma": (0, 4.5), "texture_pct": (0, 3.5),
            "isolated_specks_per_mp": (0, 250), "faint_stroke_pct": (0, 2.5), "illumination_spread": (0, 80),
            "paper_tint": (0, 22), "paper_level": (150, 255), "ink_level": (25, 200), "content_frac": (0.7, 1.0),
        },
    },
    "orient": {
        # Text mask reuses CONFIG["assess"] stroke + ruled-line settings (mm, scaled per page).
        "work_dpi": 150,                # orientation measured on a 150 DPI copy of the content box
        # 1) Axis: smear text along x and along y; horizontal text gives long flat blobs when smeared along x
        "smear_mm": 3.0,
        "smear_elongation": 4.0,        # blob counts as a "line" if length >= 4x thickness
        "axis_confident": 0.10,         # |axis score| below this = not sure which way the text runs
        # 2) Up/down: horizontal bars (Devanagari headline, T-bars) have ink hanging BELOW them when upright
        "bar_min_mm": 2.0,              # horizontal strokes at least this long count as bars
        "bar_window_mm": 2.0,           # compare ink this far above vs below each bar
        "updown_rotate_below": -0.5,    # vote <= this: rotate 180 (no upright sample went below -0.40)
        "updown_flag_below": -0.15,     # vote <= this: flag "possibly_upside_down" for review (English pages
                                        # give weak/negative votes; the vote is reliable for Hindi text)
        "min_text_px": 2000,            # fewer text pixels (at work_dpi) = too little text to decide
    },
    "borders": {
        "fill": "paper",                # "paper" = this page's paper colour (Step 2), or "white"
        "protect_mm": 1.5,              # ink within this distance inside the content box that crosses into a band is kept
        "restore_pad_mm": 0.25,         # kept strokes get this much extra margin (anti-aliased edges)
        "restore_reach_mm": 4.0,        # a kept stroke may reach at most this far into the band (descender, loop)
        "restore_max_stroke_mm2": 60,   # connected ink bigger than this is shadow/texture, not a pen stroke
        "restore_max_stroke_width_mm": 0.8,  # thicker than this (widest point) is a blob, not a pen stroke
        "restore_min_background_rel": 0.75,  # only restore strokes lying on paper-bright background (>= 75% of the
                                             # page's paper level); strokes inside a dark spine shadow are not text
        "feather_mm": 0.5,              # soft blend from the kept page into the filled band
        # Dark (spine/shadow) bands are FLATTENED, not painted over: page text often lies on the shadow.
        "shadow_smooth_mm": 0.4,        # blur speckle before estimating the shadow level
        "shadow_bg_mm": 3.8,            # background = closing wider than any stroke (after the blur)
        "shadow_max_gain": 6.0,         # background darker than paper/6 is too dark to flatten ...
        "shadow_lift": None,            # None (default, reviewed and chosen): flattened shadow only, no data loss.
                                        # (0.45, 0.80) = optional MONOTONE tone curve inside dark bands: lightness below
                                        # 45% of paper kept (text), above 80% -> paper (grain), smooth between; also
                                        # reviewed as acceptable. (0.40, 0.70) was judged too strong.
                                        # Binary "text or not" masks were tried and erased characters.
        "shadow_lift_feather_mm": 1.0,  # soft edge where the band meets the page
        "fill_unrecoverable_dark": False,  # ... and is LEFT AS ORIGINAL (False, safe: may hold white-on-black text or
                                           # text the eye can still read). True paints it with paper (cosmetic only).
        "shadow_mask_smooth_mm": 2.0,   # smooth the "unrecoverable" mask so the fill edge is not jagged
        "shadow_grain_rel": 0.8,        # after flattening, blurred level >= 80% of paper = grain, not ink -> paper
        "shadow_grain_blur_mm": 0.3,    # (0.7 / 0.4 mm erased faint text on the shadow; leftover grain -> Step 7)
        "crease_smooth_mm": 1.0,        # profile smoothing when locating the spine
        # White-on-dark print (black banners with white text) inside a dark band must not be flattened/filled
        "inverted_bg_rel": 0.35,        # pixels darker than 35% of paper are "black" ...
        "inverted_solid_mm": 4.0,       # ... a white-on-black area is one where >= 60% of a 4 mm window is black
        "inverted_solid_frac": 0.6,     #     (gaps between dark letters on paper are NOT a dark background) ...
        "inverted_contrast_rel": 0.5,   # ... and marks brighter than it by >= 50% of paper = white-on-black glyphs
                                        # (spine shadows: grey ~40-50% of paper, gaps only ~25% brighter -> no)
        "inverted_min_glyph_mm2": 0.3,  # bright marks at least this big ...
        "inverted_glyph_size_mm": (0.5, 15.0),  # ... with min side >= 0.5 mm and max side <= 15 mm ...
        "inverted_max_elongation": 6.0, # ... and not line-like (crease highlights)
        "inverted_min_glyphs": 5,       # this many glyph-like marks in a dark band -> flag "inverted_content_in_border"
        "shadow_edge_rel": 0.5,         # the shadow's outer edge = where the profile rises back above
                                        # min + 50% of (paper - min) on the edge side of the darkest line.
                                        # Area beyond it may be a neighbour-page sliver OR this page continuing
                                        # (lines written across the spine): it is flattened, never erased, and
                                        # flagged "content_beyond_spine" if it holds ink.
    },
    "deskew": {
        "work_dpi": 150,                # skew measured on a 150 DPI copy of the content box
        "max_angle": 10.0,              # search range +- degrees
        "coarse_step": 0.5, "fine_step": 0.05,  # projection search: coarse, then fine around the best
        "min_text_px": 2000,            # fewer text pixels at work_dpi = no projection estimate
        # printed / dotted lines (Hough)
        "line_min_mm": 30,              # segments at least this long
        "line_gap_mm": 1.5,             # bridge gaps (dotted fill-in lines)
        "line_votes_mm": 20,            # Hough accumulator threshold (in mm of collinear ink)
        "min_lines": 3,                 # fewer segments = no line estimate
        # decision (thresholds from the self-test on tilted sample pages)
        "lines_strong_min": 10,         # >= 10 segments and ...
        "lines_strong_mm": 1000,        # ... >= 1 m total length: median error 0.09 deg, 95% within 0.5 deg
        "projection_strong_peak": 1.1,  # text-row sharpness at best angle / at 0 deg; below this it was often wrong
        "agree_deg": 0.5,               # estimates within this agree
        "projection_conflict_deg": 1.0, # projection decides, but even weak line evidence differs by more -> uncertain
        "min_rotate_deg": 0.1,          # smaller skew is left alone
        "uncertain_flag_deg": 0.5,      # no trustworthy estimate but one suggests >= this -> flag
    },
    "illumination": {
        "apply_to_all_pages": True,     # True: every non-blank page gets the same white paper (pages that are already
                                        # white change by ~nothing); False: only pages Step 2 says need it
        "bg_dpi": 50,                   # paper colour estimated on a 50 DPI copy
        "bg_window_mm": 12.0,           # closing window: wider than strokes, dense handwriting gaps, banner bars
        "bg_smooth_mm": 12.0,           # median + Gaussian smoothing of the estimate (no halos)
        "target": 255,                  # paper becomes this (white)
        "max_gain": 1.5,                # never brighten more than this: dark areas wider than the window
                                        # (thumbprints, big stamps) are not bleached
        "min_gain": 0.95,               # and never darken by more than 5%
        "snap_start": 245,              # soft white-point: luminance from here to 255 is eased toward pure white
                                        # (was 235: faded pencil marks and very light stamps)
        "dark_object_rel": 0.55,        # estimated "paper" darker than 55% of the page's paper = an OBJECT (photo,
                                        # thumbprint, seal, black banner), not paper: the paper estimate there is
                                        # filled in from the surrounding paper so the object is not brightened
    },
    "denoise": {
        # Decided from metrics re-measured on the current image (Step 4 flattening can add grain after Step 2).
        # Level thresholds reuse CONFIG["assess"]["needs"] (denoise_mild_sigma / denoise_strong_sigma / ...).
        "nlm_h": {"mild": 6, "strong": 12},   # non-local-means strength on lightness (0-255 scale)
        "nlm_template_mm": 0.9,         # patch size
        "nlm_search_mm": 2.7,           # search window
        "chroma_median_mm": 0.6,        # median on Lab a/b removes colour speckle - applied to PAPER pixels only,
                                        # so thin coloured strokes keep their colour
        "texture_cleanup_pct": 0.5,     # faint dot-like texture above this % of the page -> set those dots to paper
                                        # (only NEUTRAL grey marks no darker than stroke_ink are ever cleared:
                                        #  coloured marks - stamps, seals, ink dots - are never erased)
        "texture_target": 255,          # paper value after Step 6
        "stroke_min_mm2": 0.5,          # ink-loss check counts only connected ink >= this (real strokes, not grain)
    },
    "sharpen": {
        # Decided from sharpness re-measured on the current image; threshold = CONFIG["assess"]["needs"]["sharpen_below"]
        "sigma_mm": 0.2,                # unsharp-mask blur radius (about a stroke width)
        "target_sharpness": 9.0,        # amount = gain x (target - sharpness) / target ...
        "gain": 3.0,
        "min_amount": 0.3, "max_amount": 1.5,   # ... clipped to this range
        "detail_threshold": 4,          # |detail| below this (grey levels) is noise/paper: not boosted
        "halo_overshoot": 8,            # result may exceed the local 3x3 min/max by at most this (no halos)
    },
    "bleed": {
        # Bleed-through = the SAME ink colour appearing as two lightness populations: dark (front) and light
        # (show-through from the back). Ink components are grouped by colour; black/grey is not used (printed text
        # and grey dotted lines always split, even on clean forms).
        "min_component_mm2": 0.3,       # ink components at least this big take part
        "groups": ["blue", "purple", "red"],  # colour groups tested (hue bands from CONFIG["assess"]["ink_hues"])
        "min_components": 100,          # a group needs this many components to judge
        "min_eta": 0.8,                 # Otsu separability (between-class / total variance) of component lightness
        "min_gap_L": 30,                # light mean - dark mean (Lab L, 0-255)
        "light_share": (0.25, 0.85),    # area share of the light population must be in this range (0.15 caught faint
                                        # real handwriting on FIR-4-2013 p5/p18)
        "lighten": 0.6,                 # blend light PIXELS of that colour 60% toward paper: show-through fades but
                                        # anything lightened stays readable if the detection was wrong (was 0.85)
        "protect_mm": 0.25,             # never touch pixels this close to dark same-colour ink (front strokes and their
                                        # soft edges) or to black/grey ink (print): bleed touches both
        "feather_mm": 0.1,
        "paper": 255,
    },
    "contrast": {
        # Faint ink is detected per MARK, not per page: every connected stroke's core (its darkest pixel) is measured.
        # Dark handwriting has cores ~30-90; light-grey printed toner / faded ink has cores ~130-215. A page whose
        # faint-core marks are a real share of its ink gets a monotone tone curve; others are left alone.
        "faint_core_min": 110,          # a mark whose core is lighter than this counts as faint
        "faint_min_mark_mm2": 0.10,     # marks smaller than this are ignored (grain, specks)
        "measure_threshold": "stroke_ink",  # marks used to MEASURE the faint level (CONFIG["assess"] black-hat level)
        "apply_threshold": "stroke_weak",   # marks the curve is APPLIED to: the weak level also reaches the faintest
                                            # printed letters (measuring with it anchored the curve near paper)
        "mark_pad_px": 1,               # the curve is applied ONLY to faint marks (+ this soft edge): paper, dark
                                        # handwriting, photos, stamps are untouched (a whole-page curve flattened
                                        # dark photos in the self-test)
        "faint_share_min": 0.10,        # apply when >= 10% of the ink area is faint marks
        "target_faint_core": 80,        # the typical faint core is mapped to this grey (clearly readable; 110 was
                                        # still light on faint printed form text)
        "knee": 240,                    # lightness >= knee (paper) is never changed
        "max_faint_slope": 3.5,         # limit on how strongly the faint..paper range is stretched (noise)
        "smooth": 9,                    # smoothing of the curve (LUT entries); keeps it monotone
    },
    "device_overrides": {},       # any device-profile key, e.g. {"num_workers": 2}
    "review": {                   # thresholds that raise review flags
        "render": {
            "min_short_side_px": 1240,  # ~150 DPI on A4 width
            "min_image_coverage": 0.95, # scan image must cover >= 95% of the page
        },
        "assess": {
            "blank_ink_pct": 0.3,       # below = blank page (later steps skip it)
            "near_blank_ink_pct": 1.0,  # below = nearly blank
            "blurry_below": 5.0,        # sharpness score (synthetic blur sigma 1.5 ~ 4-5)
            "noisy_sigma": 2.0,         # noise sigma
            "specks_per_mp": 500,       # isolated specks / megapixel
            "bleed_faint_stroke_pct": 1.0,
            "large_border_frac": 0.08,  # any border band wider than 8% of that side
            "uneven_illumination": 50,  # paper-level spread (grey levels)
        },
        "borders": {
            "text_at_border_mm2": 20,   # strokes kept inside a band larger than this -> "text_at_border"
            "large_fill_pct": 20,       # filled area above this % of the page -> "large_fill"
            "beyond_spine_ink_mm2": 20, # ink beyond the spine shadow above this -> "content_beyond_spine"
            # "inverted_content_in_border" is always raised when a dark band is left untouched
        },
        "deskew": {
            "large_skew_deg": 5.0,      # rotated by more than this -> "large_skew" (often slanted handwriting)
        },
        "illumination": {
            "ink_loss_pct": 10,         # stroke pixels dropped by more than this % -> "ink_loss"
            "colour_loss_pts": 10,      # blue/purple/red share of strokes DROPPED by more than this -> "colour_loss"
                                        # (a rise is expected: ink on yellow/pink paper regains its true colour)
            "tint_remaining": 4,        # paper chroma after -> "tint_remaining"
            "illumination_remaining": 15,  # paper-level spread after -> "illumination_remaining"
            "gain_capped_pct": 1.0,     # more than this % of the page hit max_gain -> "gain_capped"
            "dark_object_pct": 0.5,     # dark objects protected on more than this % of the page -> "dark_object_kept" (info)
        },
        "denoise": {
            "ink_loss_pct": 8,          # real-stroke pixels (>= stroke_min_mm2) dropped by more than this % -> "ink_loss"
            "noise_remaining_sigma": 2.0,   # noise sigma after -> "noise_remaining"
            "specks_remaining_per_mp": 100, # isolated specks/MP after -> "specks_remaining"
        },
        "sharpen": {
            "still_blurry_below": 5.0,  # sharpness after -> "still_blurry"
            "noise_rise_sigma": 0.5,    # noise sigma rose by more than this -> "noise_amplified"
            "stroke_change_pct": 10,    # real-stroke pixels changed by more than this % -> "ink_change"
        },
        "bleed": {
            "stroke_change_pct": 5,     # dark-population (front) stroke pixels changed by more than this % -> "ink_change"
        },
        "contrast": {
            "still_faint_core": 170,    # typical faint core still lighter than this after -> "still_faint"
            "noise_rise_sigma": 0.5,    # noise sigma rose by more than this -> "noise_amplified"
            "texture_rise_pct": 0.5,    # texture % rose by more than this -> "noise_amplified"
        },
        "output": {
            "bw_ink_recall": 0.85,      # share of stroke CORES (black-hat > stroke_strong, in real strokes) that are black
                                        # in the B&W; below -> "bw_ink_loss" (soft stroke edges are not counted)
            "bw_noise_pct": 1.0,        # black pixels far from any stroke, % of page; above -> "bw_noise"
        },
    },
}
DEFAULT_CONFIG = copy.deepcopy(CONFIG)

# Device profile (backends, workers). Module-level dicts so that notebook code sharing them stays in sync.
PROFILE: dict = {}


def set_device(overrides: dict | None = None, device: str | None = None) -> dict:
    """(Re)detect the device and apply its OpenCV settings. device: None (auto), "mac", "cuda" or "cpu"."""
    old = os.environ.get("FIR_DEVICE")
    try:
        if device:
            os.environ["FIR_DEVICE"] = device
        prof = get_device_profile(overrides or CONFIG.get("device_overrides") or {})
    finally:
        if device:
            if old is None:
                os.environ.pop("FIR_DEVICE", None)
            else:
                os.environ["FIR_DEVICE"] = old
    PROFILE.clear()
    PROFILE.update(prof)
    apply_profile(PROFILE)
    return PROFILE


set_device()


# ====================================================================================================
# Page records and shared helpers
# ====================================================================================================
# ---------- Page record helpers (used by every step) ----------

def new_page_record(doc_name: str, doc_path: Path, page_index: int, page, n_pages: int = 0) -> dict:
    w_pt, h_pt = page.get_size()
    return {
        "n_pages": n_pages,
        "doc": doc_name,
        "doc_path": str(doc_path),
        "page_no": page_index + 1,          # 1-based, as people count pages
        "page_index": page_index,           # 0-based, for pypdfium2
        "size_pt": (round(w_pt, 1), round(h_pt, 1)),
        "pdf_orientation": "landscape" if w_pt > h_pt else "portrait",
        "images": {},      # stage name -> np.ndarray (e.g. "original", "clean_color")
        "metrics": {},     # step name -> dict of numbers
        "actions": [],     # (step, what was done)
        "flags": [],       # dicts: step, code, message  -> drives the review filter
        "dpi": None,       # set in Step 1 (estimated or CONFIG override); all physical sizes use it
    }


def add_metrics(record: dict, step: str, **metrics) -> None:
    record["metrics"].setdefault(step, {}).update(metrics)


def add_action(record: dict, step: str, message: str) -> None:
    record["actions"].append((step, message))


def add_flag(record: dict, step: str, code: str, message: str) -> None:
    record["flags"].append({"step": step, "code": code, "message": message})


def mm_to_px(mm: float, dpi: float, odd: bool = False, min_px: int = 1) -> int:
    """Physical size -> pixels at this page's DPI (odd=True for kernel sizes)."""
    px = max(min_px, int(round(mm * dpi / 25.4)))
    return px + 1 if odd and px % 2 == 0 else px


def page_label(record: dict) -> str:
    return f"{record['doc']} p{record['page_no']}"


def run_parallel(fn, records: list[dict], desc: str) -> float:
    """Run fn(record) for every record on PROFILE['num_workers'] threads (OpenCV/NumPy release the GIL).
    Returns elapsed seconds. Errors are not swallowed."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PROFILE["num_workers"]) as pool:
        list(pool.map(fn, records))
    elapsed = time.perf_counter() - t0
    print(f"{desc}: {len(records)} pages in {elapsed:.2f}s ({PROFILE['num_workers']} workers)")
    return elapsed


# ====================================================================================================
# Step 1: render
# ====================================================================================================
STEP1 = "step1_render"


def find_page_scan(page) -> tuple:
    """Return (image_object, info) if the page is one full-page, axis-aligned scan image, else (None, info)."""
    objects = list(page.get_objects())
    images = [o for o in objects if o.type == pdfium_raw.FPDF_PAGEOBJ_IMAGE]
    w_pt, h_pt = page.get_size()
    info = {"n_objects": len(objects), "n_images": len(images), "pdf_rotation": page.get_rotation(), "coverage": 0.0}
    if len(images) != 1:
        return None, info
    a, b, c, d, _, _ = images[0].get_matrix().get()
    info["coverage"] = round(float(abs(a * d) / (w_pt * h_pt)), 3)
    axis_aligned = b == 0 and c == 0 and a > 0 and d > 0
    if len(objects) == 1 and axis_aligned and info["pdf_rotation"] == 0:
        return images[0], info
    return None, info


def render_page(page, record: dict, cfg: dict, review: dict, max_scale: float) -> np.ndarray:
    t0 = time.perf_counter()
    scan, info = find_page_scan(page)

    if scan is not None:
        record["original_bytes"] = (bytes(scan.get_data(decode_simple=False))  # exact scan, saved as {doc}_{page}_original.jpg
                                    if scan.get_filters() == ["DCTDecode"] else None)
        bitmap = scan.get_bitmap(render=False)          # decoded JPEG pixels, no resampling
        img = np.array(bitmap.to_numpy(), copy=True)    # copy out of pdfium memory (to_numpy is a view)
        if bitmap.format in (pdfium_raw.FPDFBitmap_BGRA, pdfium_raw.FPDFBitmap_BGRx):
            img = img[..., :3]
        elif bitmap.format == pdfium_raw.FPDFBitmap_Gray:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        method = "embedded_image"
        codec = "/".join(scan.get_filters()) or "raw"
        add_action(record, STEP1, f"extracted embedded {codec} scan at native {img.shape[1]}x{img.shape[0]} px")
    else:
        scale = cfg["fallback_dpi"] / 72.0
        bitmap = page.render(scale=scale, rev_byteorder=False)
        img = np.array(bitmap.to_numpy()[..., :3], copy=True)  # BGR(x) -> BGR, copied out of pdfium memory
        bitmap.close()
        method = "rendered"
        codec = None
        record["original_bytes"] = None
        add_action(record, STEP1, f"rendered page at {cfg['fallback_dpi']} DPI (not a single full-page scan)")
        add_flag(record, STEP1, "not_single_scan",
                 f"{info['n_images']} image(s), {info['n_objects']} object(s): rendered instead of extracted")

    img = np.ascontiguousarray(img)
    native_h, native_w = img.shape[:2]

    upscale = min(cfg["upscale"], max_scale)
    if upscale > 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        add_action(record, STEP1, f"upscaled {upscale}x to {img.shape[1]}x{img.shape[0]} px")

    # Colour content on a 1/4-size copy. Chroma = max - min channel per pixel.
    #   color_ink_pct : % of strongly coloured pixels (blue ink, stamps, tinted paper)
    #   paper_tint    : median chroma of light (paper) pixels; 0 = neutral white/grey paper
    small = cv2.resize(img, None, fx=cfg["color_scale"], fy=cfg["color_scale"], interpolation=cv2.INTER_AREA)
    chroma = small.max(axis=2).astype(np.int16) - small.min(axis=2)
    color_ink_pct = float(np.mean(chroma > cfg["color_chroma_min"]) * 100)
    paper = small.min(axis=2) > cfg["color_paper_min"]
    paper_tint = float(np.median(chroma[paper])) if paper.any() else 0.0
    greyscale_scan = color_ink_pct < cfg["greyscale_max_ink_pct"] and paper_tint < cfg["greyscale_max_tint"]

    short_side = min(native_h, native_w)
    est_dpi = short_side / cfg["assumed_paper_width_in"]
    record["dpi"] = float(cfg["dpi_override"] or est_dpi) * upscale   # may be replaced by the document median below
    add_metrics(
        record, STEP1,
        method=method,
        codec=codec,
        native_px=(native_w, native_h),
        output_px=(img.shape[1], img.shape[0]),
        upscale=upscale,
        orientation="landscape" if native_w > native_h else "portrait",
        est_dpi=round(est_dpi),
        dpi_used=round(record["dpi"]),
        color_ink_pct=round(color_ink_pct, 2),
        paper_tint=round(paper_tint, 1),
        greyscale_scan=greyscale_scan,
        time_ms=round((time.perf_counter() - t0) * 1000, 1),
        **info,
    )

    if short_side < review["min_short_side_px"]:
        add_flag(record, STEP1, "low_resolution",
                 f"short side {short_side}px (~{est_dpi:.0f} DPI) < {review['min_short_side_px']}px")
    if info["pdf_rotation"] != 0:
        add_flag(record, STEP1, "page_rotation_set", f"PDF /Rotate = {info['pdf_rotation']}°")
    if info["n_images"] == 1 and info["coverage"] < review["min_image_coverage"]:
        add_flag(record, STEP1, "partial_image_coverage", f"scan covers {info['coverage']:.0%} of the page")
    return img


# ====================================================================================================
# Step 2: assess (measure only)
# ====================================================================================================
STEP2 = "step2_assess"
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _border_lines(dark_frac, pale, has_content, max_dark: int, max_pale: int, dark_thr: float,
                  min_dark: int, min_pale: int, grow_frac: float, grow_max: int) -> tuple:
    """Profiles ordered edge -> inward. Returns (lines from the edge that are border, kind).
    Dark strips may sit inside the edge (neighbour-page text outside a spine), so the search skips over content.
    Pale strips must be reached from the edge without crossing any line that has content (text)."""
    dark_inner, run = 0, 0
    for i in range(max_dark):                        # dark strip anywhere in the outer band
        run = run + 1 if dark_frac[i] >= dark_thr else 0
        if run >= min_dark:
            dark_inner = i + 1
    if dark_inner:                                   # shadow edge: grow inward while still partly dark
        grown = 0
        while dark_inner < len(dark_frac) and grown < grow_max and dark_frac[dark_inner] >= grow_frac:
            dark_inner += 1
            grown += 1
    pale_inner, run = 0, 0
    for i in range(max_pale):                        # pale, uniform strip near the edge
        if has_content[i] and not pale[i]:
            break                                    # reached text: the pale band cannot extend past it
        run = run + 1 if pale[i] else 0
        if run >= min_pale:
            pale_inner = i + 1
    kind = {(True, True): "dark+pale", (True, False): "dark", (False, True): "pale"}.get(
        (dark_inner > 0, pale_inner > 0))
    return max(dark_inner, pale_inner), kind


def detect_borders(img: np.ndarray, cfg: dict, dpi: float) -> tuple:
    """Dark (spine/scanner shadow) and pale (scanner bed) bands per side, on a 1/q copy."""
    q = cfg["work_scale"]
    sdpi = dpi / q
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // q, h // q), interpolation=cv2.INTER_AREA)
    sg = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    sh, sw = sg.shape
    center = lab[sh // 4: 3 * sh // 4, sw // 4: 3 * sw // 4].reshape(-1, 3)
    ref = center[center[:, 0] >= np.percentile(center[:, 0], cfg["paper_ref_percentile"])]
    paper_lab = np.median(ref, axis=0)
    paper_grey = float(np.median(cv2.cvtColor(small[sh // 4: 3 * sh // 4, sw // 4: 3 * sw // 4],
                                              cv2.COLOR_BGR2GRAY)))
    de = np.sqrt(((lab - paper_lab) ** 2).sum(axis=2))
    dark = sg < cfg["border_dark_rel"] * paper_grey               # relative to this page's paper
    min_dark = mm_to_px(cfg["border_dark_min_mm"], sdpi)
    min_pale = mm_to_px(cfg["border_pale_min_mm"], sdpi)
    light = lab[..., 0]

    borders = {}
    for side in ("left", "right", "top", "bottom"):
        axis = 0 if side in ("left", "right") else 1        # average down columns / along rows
        dark_frac, mean_de, l_std = dark.mean(axis), de.mean(axis), light.std(axis)
        if side in ("right", "bottom"):
            dark_frac, mean_de, l_std = dark_frac[::-1], mean_de[::-1], l_std[::-1]
        pale = (mean_de >= cfg["border_pale_de"]) & (l_std < cfg["border_pale_std"])
        has_content = l_std >= cfg["border_pale_std"]
        n = len(dark_frac)
        lines, kind = _border_lines(dark_frac, pale, has_content, int(n * cfg["border_dark_search"]),
                                    int(n * cfg["border_pale_search"]), cfg["border_dark_frac"], min_dark, min_pale,
                                    cfg["border_dark_grow_frac"], mm_to_px(cfg["border_dark_grow_max_mm"], sdpi))
        borders[side] = {"px": int(lines * q), "kind": kind}
    return borders, paper_lab


def assess_page(img: np.ndarray, cfg: dict, dpi: float) -> dict:
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    px = lambda mm, odd=False: mm_to_px(mm, dpi, odd)          # physical size -> pixels on this page
    borders, _ = detect_borders(img, cfg, dpi)
    x0, x1 = borders["left"]["px"], w - borders["right"]["px"]
    y0, y1 = borders["top"]["px"], h - borders["bottom"]["px"]
    im = img[y0:y1, x0:x1]
    # Measure at the reference DPI: sharpness, noise and speck size are per-pixel properties
    ref = cfg["reference_dpi"]
    if abs(dpi / ref - 1) > 0.05:
        f = ref / dpi
        im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        dpi = ref
        px = lambda mm, odd=False: mm_to_px(mm, dpi, odd)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    area = g.size

    # Strokes: black-hat = how much darker than the local (15 px) background
    sk = px(cfg["stroke_width_max_mm"], odd=True)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    weak, ink, strong = bh > cfg["stroke_weak"], bh > cfg["stroke_ink"], bh > cfg["stroke_strong"]
    ink_px = int(ink.sum())

    # Sharpness: high-frequency energy on stroke edges relative to stroke contrast
    bk = px(cfg["sharp_band_mm"], odd=True)
    band = cv2.dilate(ink.astype(np.uint8), np.ones((bk, bk), np.uint8)).astype(bool)
    contrast = float(np.percentile(bh[ink], cfg["contrast_percentile"])) if ink_px else 0.0
    hf = cv2.absdiff(g, cv2.GaussianBlur(g, (0, 0), cfg["sharp_sigma_mm"] * dpi / 25.4))
    sharpness = float(hf[band].mean()) / max(contrast, 1.0) * 100 if ink_px else 0.0

    # Noise: Immerkaer estimate on flat pixels (exclude the 10% strongest gradients)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = grad < np.percentile(grad, cfg["noise_flat_percentile"])
    lap = np.abs(cv2.filter2D(g.astype(np.float32), -1, _IMMERKAER))
    noise_sigma = float(np.sqrt(np.pi / 2) * lap[flat].mean() / 6) if flat.any() else 0.0

    # Faint marks not near real strokes, ruled/dotted lines removed -> stroke-shaped (bleed) vs dots (texture)
    wk = weak.astype(np.uint8)
    gap, run_len, pad = px(cfg["rule_line_gap_mm"], odd=True), px(cfg["rule_line_min_mm"], odd=True), px(cfg["rule_line_pad_mm"], odd=True)
    h_lines = cv2.morphologyEx(cv2.dilate(wk, np.ones((1, gap), np.uint8)), cv2.MORPH_OPEN, np.ones((1, run_len), np.uint8))
    v_lines = cv2.morphologyEx(cv2.dilate(wk, np.ones((gap, 1), np.uint8)), cv2.MORPH_OPEN, np.ones((run_len, 1), np.uint8))
    lines = cv2.dilate(h_lines | v_lines, np.ones((pad, pad), np.uint8)).astype(bool)
    nk = px(cfg["near_ink_mm"], odd=True)
    near_strong = cv2.dilate(strong.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nk, nk))).astype(bool)
    faint = (weak & ~near_strong & ~lines).astype(np.uint8)
    _, _, st, _ = cv2.connectedComponentsWithStats(faint, connectivity=8)
    fa, fw, fh = st[1:, cv2.CC_STAT_AREA], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    mm2_px = (dpi / 25.4) ** 2                                 # pixels per mm^2
    stroke_like = ((fa >= cfg["faint_stroke_min_area_mm2"] * mm2_px)
                   & (np.minimum(fw, fh) >= px(cfg["faint_stroke_min_size_mm"]))
                   & (fa / np.maximum(1, fw * fh) < cfg["faint_stroke_max_fill"]))
    faint_stroke_pct = float(fa[stroke_like].sum()) / area * 100
    texture_pct = float(fa[~stroke_like].sum()) / area * 100

    # Isolated specks: ink blobs <= 6 px with no other ink in a 17x17 window
    _, _, st2, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    ia = st2[1:, cv2.CC_STAT_AREA]
    win = px(cfg["speck_isolation_mm"], odd=True)
    box = cv2.boxFilter(ink.astype(np.float32), -1, (win, win), normalize=False)
    cx = np.clip(np.rint(cen[1:, 0]).astype(int), 0, g.shape[1] - 1)
    cy = np.clip(np.rint(cen[1:, 1]).astype(int), 0, g.shape[0] - 1)
    specks = int(((ia <= cfg["speck_max_area_mm2"] * mm2_px) & (box[cy, cx] <= ia + 0.5)).sum())

    # Paper level / illumination on a 1/q copy (closing removes strokes)
    q = cfg["work_scale"]
    sg = cv2.resize(g, (max(1, g.shape[1] // q), max(1, g.shape[0] // q)), interpolation=cv2.INTER_AREA)
    ck = mm_to_px(cfg["illum_close_mm"], dpi / q, odd=True)
    mk = mm_to_px(cfg["illum_median_mm"], dpi / q, odd=True, min_px=3)
    bg = cv2.medianBlur(cv2.morphologyEx(sg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))), mk)
    lo, hi = cfg["illum_percentiles"]
    illumination_spread = float(np.percentile(bg, hi) - np.percentile(bg, lo))

    # Paper colour (pixels away from any mark) and ink levels/colours
    pk = px(cfg["paper_pad_mm"], odd=True)
    paper = ~cv2.dilate(wk, np.ones((pk, pk), np.uint8)).astype(bool)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    paper_lab = np.median(lab[paper], axis=0) if paper.any() else np.array([255, 128, 128])
    paper_tint = float(np.hypot(paper_lab[1] - 128.0, paper_lab[2] - 128.0))
    paper_level = float(np.median(g[paper])) if paper.any() else 255.0
    ink_level = float(np.median(g[ink])) if ink_px else paper_level

    chroma = im.max(axis=2).astype(np.int16) - im.min(axis=2)
    hue = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 0]
    colored = ink & (chroma >= cfg["ink_chroma_min"])
    n_ink = max(1, ink_px)
    def hue_band(lo_hi):                                      # red wraps around 180 -> lo > hi
        lo_h, hi_h = lo_hi
        return (hue >= lo_h) & (hue < hi_h) if lo_h < hi_h else (hue >= lo_h) | (hue < hi_h)
    hues = cfg["ink_hues"]
    blue = int((colored & hue_band(hues["blue"])).sum())
    purple = int((colored & hue_band(hues["purple"])).sum())
    red = int((colored & hue_band(hues["red"])).sum())
    ink_colors = {
        "blue": round(blue / n_ink * 100, 1), "purple": round(purple / n_ink * 100, 1),
        "red": round(red / n_ink * 100, 1), "other_colour": round((int(colored.sum()) - blue - purple - red) / n_ink * 100, 1),
        "black_grey": round((ink_px - int(colored.sum())) / n_ink * 100, 1),
    }

    return {
        "borders": borders,
        "content_box": (int(x0), int(y0), int(x1), int(y1)),
        "content_frac": round((x1 - x0) * (y1 - y0) / (h * w), 3),
        "measured_at_dpi": round(dpi),
        "ink_pct": round(ink_px / area * 100, 2),
        "sharpness": round(sharpness, 2),
        "noise_sigma": round(noise_sigma, 2),
        "texture_pct": round(texture_pct, 2),
        "isolated_specks_per_mp": round(specks / (area / 1e6)),
        "faint_stroke_pct": round(faint_stroke_pct, 2),
        "illumination_spread": round(illumination_spread, 1),
        "paper_level": round(paper_level),
        "paper_lab": [round(float(v), 1) for v in paper_lab],
        "paper_tint": round(paper_tint, 1),
        "ink_level": round(ink_level),
        "ink_contrast": round(paper_level - ink_level),
        "ink_colors": ink_colors,
        "time_ms": round((time.perf_counter() - t0) * 1000),
    }


def decide_needs(rec: dict, m: dict, need: dict, review: dict, ranges: dict) -> dict:
    """Turn metrics into per-step decisions + review flags."""
    h, w = rec["images"]["original"].shape[:2]
    blank = m["ink_pct"] < review["blank_ink_pct"]
    if m["noise_sigma"] >= need["denoise_strong_sigma"]:
        denoise = "strong"
    elif m["noise_sigma"] >= need["denoise_mild_sigma"] or m["texture_pct"] >= need["denoise_texture_pct"]:
        denoise = "mild"
    else:
        denoise = "none"
    needs = {
        "skip_page": blank,
        "border_cleanup": any(b["px"] > 0 for b in m["borders"].values()),
        "illumination": m["illumination_spread"] >= need["illumination_spread"],
        "tint_removal": m["paper_tint"] >= need["paper_tint"],
        "denoise": denoise,
        "speck_removal": m["isolated_specks_per_mp"] >= need["speck_removal_per_mp"],
        "sharpen": (not blank) and m["sharpness"] < need["sharpen_below"],
        "bleed_suppression": m["faint_stroke_pct"] >= need["bleed_faint_stroke_pct"],
        "contrast": (not blank) and m["ink_level"] >= need["faint_ink_level"],
    }

    if blank:
        add_flag(rec, STEP2, "blank", f"ink {m['ink_pct']}% < {review['blank_ink_pct']}%: later steps skip this page")
    elif m["ink_pct"] < review["near_blank_ink_pct"]:
        add_flag(rec, STEP2, "near_blank", f"ink {m['ink_pct']}% < {review['near_blank_ink_pct']}%")
    if not blank and m["sharpness"] < review["blurry_below"]:
        add_flag(rec, STEP2, "blurry", f"sharpness {m['sharpness']} < {review['blurry_below']}")
    if m["noise_sigma"] >= review["noisy_sigma"]:
        add_flag(rec, STEP2, "noisy", f"noise sigma {m['noise_sigma']} >= {review['noisy_sigma']}")
    if m["isolated_specks_per_mp"] >= review["specks_per_mp"]:
        add_flag(rec, STEP2, "many_specks", f"{m['isolated_specks_per_mp']} isolated specks/MP")
    if m["faint_stroke_pct"] >= review["bleed_faint_stroke_pct"]:
        add_flag(rec, STEP2, "bleed_through_suspected", f"faint stroke-like marks {m['faint_stroke_pct']}% of page")
    big = [s for s, b in m["borders"].items()
           if b["px"] >= review["large_border_frac"] * (w if s in ("left", "right") else h)]
    if big:
        add_flag(rec, STEP2, "large_border",
                 ", ".join(f"{s} {m['borders'][s]['px']}px ({m['borders'][s]['kind']})" for s in big))
    if m["illumination_spread"] >= review["uneven_illumination"]:
        add_flag(rec, STEP2, "uneven_illumination", f"paper level spread {m['illumination_spread']}")
    # Unknown-issue guard: anything outside the range seen on the sample set gets a human look
    for key, (lo, hi) in ranges.items():
        if key == "ink_pct" and blank:
            continue
        if not lo <= m[key] <= hi:
            add_flag(rec, STEP2, f"unusual_{key}", f"{key}={m[key]} outside sample range [{lo}, {hi}]: possible new issue")
    return needs


def step2_assess(rec: dict) -> None:
    if rec["metrics"].get(STEP2):              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP2]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP2]
    m = assess_page(rec["images"]["original"], CONFIG["assess"], rec["dpi"])
    rec["metrics"][STEP2] = m
    rec["needs"] = decide_needs(rec, m, CONFIG["assess"]["needs"], CONFIG["review"]["assess"],
                                CONFIG["assess"]["expected_ranges"])
    todo = [k if v is True else f"{k}={v}" for k, v in rec["needs"].items() if v not in (False, "none")]
    add_action(rec, STEP2, "measured only; needs: " + (", ".join(todo) or "nothing"))


# ====================================================================================================
# Step 3: orientation
# ====================================================================================================
STEP3 = "step3_orient"
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(img: np.ndarray, deg_cw: int) -> np.ndarray:
    return img if deg_cw == 0 else cv2.rotate(img, ROTATIONS[deg_cw])


def orientation_text_mask(img: np.ndarray, box: tuple, dpi: float, acfg: dict, work_dpi: float) -> np.ndarray:
    """Binary stroke mask of the content box at work_dpi, ruled/dotted lines removed (same settings as Step 2)."""
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    f = work_dpi / dpi
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    px = lambda mm, odd=False: mm_to_px(mm, work_dpi, odd)
    k = px(acfg["stroke_width_max_mm"], odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ink = (bh > acfg["stroke_ink"]).astype(np.uint8)
    gap, run_len = px(acfg["rule_line_gap_mm"], odd=True), px(acfg["rule_line_min_mm"], odd=True)
    h_lines = cv2.morphologyEx(cv2.dilate(ink, np.ones((1, gap), np.uint8)), cv2.MORPH_OPEN, np.ones((1, run_len), np.uint8))
    v_lines = cv2.morphologyEx(cv2.dilate(ink, np.ones((gap, 1), np.uint8)), cv2.MORPH_OPEN, np.ones((run_len, 1), np.uint8))
    pad = px(acfg["rule_line_pad_mm"], odd=True)
    lines = cv2.dilate(h_lines | v_lines, np.ones((pad, pad), np.uint8))
    return ink & (1 - lines)


def axis_score(mask: np.ndarray, dpi: float, cfg: dict) -> float:
    """> 0: text lines run horizontally; < 0: vertically."""
    k = mm_to_px(cfg["smear_mm"], dpi)
    share = {}
    for name, kernel in (("h", np.ones((1, k), np.uint8)), ("v", np.ones((k, 1), np.uint8))):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, _, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        w, h, a = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
        long_blob = (w >= cfg["smear_elongation"] * h) if name == "h" else (h >= cfg["smear_elongation"] * w)
        share[name] = float(a[long_blob].sum()) / max(1, int(a.sum()))
    return share["h"] - share["v"]


def bar_vote(mask: np.ndarray, dpi: float, cfg: dict) -> float:
    """(ink below bars - ink above bars) / total, for horizontal bars. > 0 = looks upright."""
    bars = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, mm_to_px(cfg["bar_min_mm"], dpi)), np.uint8))
    if not bars.any():
        return 0.0
    d = mm_to_px(cfg["bar_window_mm"], dpi)
    rest = (mask & (1 - cv2.dilate(bars, np.ones((3, 1), np.uint8)))).astype(np.float32)
    k_up = np.zeros((2 * d + 1, 1), np.float32); k_up[:d] = 1          # rows y-d .. y-1
    k_dn = np.zeros((2 * d + 1, 1), np.float32); k_dn[d + 1:] = 1      # rows y+1 .. y+d
    above = cv2.filter2D(rest, -1, k_up, borderType=cv2.BORDER_CONSTANT)[bars.astype(bool)].sum()
    below = cv2.filter2D(rest, -1, k_dn, borderType=cv2.BORDER_CONSTANT)[bars.astype(bool)].sum()
    return float((below - above) / max(1.0, below + above))


def updown_score(mask: np.ndarray, dpi: float, cfg: dict) -> float:
    """vote(as is) - vote(turned 180). > 0 = upright, < 0 = upside down."""
    return bar_vote(mask, dpi, cfg) - bar_vote(cv2.rotate(mask, cv2.ROTATE_180), dpi, cfg)


def detect_orientation(mask: np.ndarray, dpi: float, cfg: dict) -> dict:
    """Return the clockwise turn that makes the text upright, with evidence and a confidence label."""
    if int(mask.sum()) < cfg["min_text_px"]:
        return {"rotation": 0, "decision": "too_little_text", "axis_score": None, "updown_score": None}
    axis = axis_score(mask, dpi, cfg)
    if axis <= -cfg["axis_confident"]:                       # clearly sideways: choose 90 or 270 by up/down vote
        ud_cw = updown_score(cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE), dpi, cfg)
        rotation = 90 if ud_cw >= 0 else 270
        sure = abs(ud_cw) >= abs(cfg["updown_flag_below"])
        return {"rotation": rotation, "decision": "sideways" if sure else "sideways_direction_unsure",
                "axis_score": round(axis, 3), "updown_score": round(abs(ud_cw), 3)}
    ud = updown_score(mask, dpi, cfg)
    if axis < 0:                                             # leaning vertical but weak: don't turn
        return {"rotation": 0, "decision": "axis_unsure", "axis_score": round(axis, 3), "updown_score": round(ud, 3)}
    if ud <= cfg["updown_rotate_below"]:
        return {"rotation": 180, "decision": "upside_down", "axis_score": round(axis, 3), "updown_score": round(ud, 3)}
    decision = "possibly_upside_down" if ud <= cfg["updown_flag_below"] else "upright"
    return {"rotation": 0, "decision": decision, "axis_score": round(axis, 3), "updown_score": round(ud, 3)}


def rotate_box(box: tuple, w: int, h: int, deg_cw: int) -> tuple:
    """Content box (x0, y0, x1, y1) in a w x h image -> same region after a clockwise turn."""
    x0, y0, x1, y1 = box
    if deg_cw == 90:
        return (h - y1, x0, h - y0, x1)
    if deg_cw == 180:
        return (w - x1, h - y1, w - x0, h - y0)
    if deg_cw == 270:
        return (y0, w - x1, y1, w - x0)
    return box


def rotate_borders(borders: dict, deg_cw: int) -> dict:
    order = ["top", "right", "bottom", "left"]                 # clockwise order of sides
    shift = deg_cw // 90
    return {order[(order.index(side) + shift) % 4]: v for side, v in borders.items()}


def step3_orient(rec: dict) -> None:
    if rec["metrics"].get(STEP3):                              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP3]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP3]
    cfg, img = CONFIG["orient"], rec["images"]["original"]
    s2 = rec["metrics"][STEP2]
    h, w = img.shape[:2]

    if rec["needs"]["skip_page"]:
        result = {"rotation": 0, "decision": "blank_not_checked", "axis_score": None, "updown_score": None}
    else:
        mask = orientation_text_mask(img, s2["content_box"], rec["dpi"], CONFIG["assess"], cfg["work_dpi"])
        result = detect_orientation(mask, cfg["work_dpi"], cfg)

    deg = result["rotation"]
    rec["images"]["oriented"] = rotate(img, deg)              # no copy when deg == 0
    oh, ow = rec["images"]["oriented"].shape[:2]
    rec["metrics"][STEP3] = {
        **result,
        "content_box": rotate_box(s2["content_box"], w, h, deg),
        "borders": rotate_borders(s2["borders"], deg),
        "orientation_before": "landscape" if w > h else "portrait",
        "orientation_after": "landscape" if ow > oh else "portrait",
    }

    if deg:
        add_action(rec, STEP3, f"turned {deg}° clockwise ({result['decision']}, axis {result['axis_score']}, "
                               f"up/down {result['updown_score']})")
        add_flag(rec, STEP3, "rotated", f"turned {deg}° clockwise: confirm the page now reads upright")
    else:
        add_action(rec, STEP3, f"kept as is ({result['decision']})")
    if result["decision"] == "sideways_direction_unsure":
        add_flag(rec, STEP3, "orientation_uncertain", "text is sideways but 90° vs 270° is unsure: may be upside down now")
    elif result["decision"] == "axis_unsure":
        add_flag(rec, STEP3, "orientation_uncertain", f"text direction unclear (axis score {result['axis_score']})")
    elif result["decision"] == "possibly_upside_down":
        add_flag(rec, STEP3, "possibly_upside_down",
                 f"weak upside-down evidence (up/down {result['updown_score']}): not turned")
    elif result["decision"] == "too_little_text":
        add_flag(rec, STEP3, "orientation_unchecked", "too little text to judge orientation")


# ====================================================================================================
# Step 4: borders
# ====================================================================================================
STEP4 = "step4_borders"


def paper_bgr(paper_lab) -> np.ndarray:
    lab = np.uint8([[paper_lab]])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)


def dark_band_split(g: np.ndarray, borders: dict, dpi: float, smooth_mm: float, edge_rel: float) -> tuple:
    """For each dark side, find where the shadow starts on the edge side of its darkest line.
    Returns (outer mask: edge..shadow start = neighbour page / beyond the spine, to fill;
             inner mask: shadow start..band edge, to flatten so text printed on the shadow survives)."""
    h, w = g.shape
    outer, inner = np.zeros((h, w), bool), np.zeros((h, w), bool)
    sig = max(1.0, smooth_mm * dpi / 25.4)
    for side, b in borders.items():
        if not b["px"] or "dark" not in (b["kind"] or ""):
            continue
        n = b["px"]
        if side in ("left", "right"):
            prof = (g[:, :n] if side == "left" else g[:, w - n:][:, ::-1]).mean(axis=0)
        else:
            prof = (g[:n, :] if side == "top" else g[h - n:, :][::-1, :]).mean(axis=1)
        prof = cv2.GaussianBlur(prof.astype(np.float32).reshape(1, -1), (0, 0), sig).ravel()
        darkest = int(np.argmin(prof))
        paper_lvl = float(np.percentile(g, 90))
        level = prof[darkest] + edge_rel * (paper_lvl - prof[darkest])
        crease = darkest                                       # walk from the darkest line toward the edge
        while crease > 0 and prof[crease - 1] <= level:
            crease -= 1                                        # lines [0, crease) are beyond the shadow -> fill
        o, i = np.zeros((h, w), bool), np.zeros((h, w), bool)
        if side == "left":
            o[:, :crease] = True; i[:, crease:n] = True
        elif side == "right":
            o[:, w - crease:] = True; i[:, w - n:w - crease] = True
        elif side == "top":
            o[:crease, :] = True; i[crease:n, :] = True
        else:
            o[h - crease:, :] = True; i[h - n:h - crease, :] = True
        outer |= o
        inner |= i
    return outer, inner & ~outer


def dark_band_mask(h: int, w: int, borders: dict) -> np.ndarray:
    """Union of the dark (spine/shadow) border bands, from the edges."""
    dark = np.zeros((h, w), bool)
    for side, b in borders.items():
        if not b["px"] or "dark" not in (b["kind"] or ""):
            continue
        n = b["px"]
        if side == "left":
            dark[:, :n] = True
        elif side == "right":
            dark[:, w - n:] = True
        elif side == "top":
            dark[:n, :] = True
        else:
            dark[h - n:, :] = True
    return dark


def shadow_lift(img: np.ndarray, region: np.ndarray, paper_bgr_v: np.ndarray, dpi: float, lift: tuple,
                feather_mm: float) -> np.ndarray:
    """Monotone curve on Lab L inside region: L <= a*paper unchanged, L >= b*paper -> paper, smoothstep between."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    paper_l = float(cv2.cvtColor(paper_bgr_v.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0, 0])
    a, b = lift[0] * paper_l, lift[1] * paper_l
    x = np.arange(256, dtype=np.float32)
    t = np.clip((x - a) / max(1.0, b - a), 0, 1)
    t = t * t * (3 - 2 * t)
    lut = np.clip(x + t * (paper_l - x) + 0.5, 0, 255).astype(np.uint8)
    w = cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(1.0, feather_mm * dpi / 25.4))
    L = lab[..., 0].astype(np.float32)
    lab[..., 0] = np.clip(L * (1 - w) + cv2.LUT(lab[..., 0], lut).astype(np.float32) * w + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def inverted_glyphs(g: np.ndarray, region: np.ndarray, dpi: float, cfg: dict, acfg: dict, paper_level: float) -> tuple:
    """Bright, glyph-shaped marks on a dark background inside region (white-on-black print). Returns (count, mm2)."""
    if not region.any():
        return 0, 0.0
    sk = mm_to_px(acfg["stroke_width_max_mm"], dpi, odd=True)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    black = (g < cfg["inverted_bg_rel"] * paper_level).astype(np.float32)
    win = mm_to_px(cfg["inverted_solid_mm"], dpi, odd=True)
    dark_bg = cv2.boxFilter(black, -1, (win, win)) >= cfg["inverted_solid_frac"]   # solid dark surroundings
    bright = ((tophat > cfg["inverted_contrast_rel"] * paper_level) & dark_bg & region).astype(np.uint8)
    _, _, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    mm = 25.4 / dpi
    w, h, a = st[1:, cv2.CC_STAT_WIDTH] * mm, st[1:, cv2.CC_STAT_HEIGHT] * mm, st[1:, cv2.CC_STAT_AREA] * mm * mm
    lo, hi = cfg["inverted_glyph_size_mm"]
    short, long_ = np.minimum(w, h), np.maximum(w, h)
    glyph = ((a >= cfg["inverted_min_glyph_mm2"]) & (short >= lo) & (long_ <= hi)
             & (long_ / np.maximum(short, 1e-6) <= cfg["inverted_max_elongation"]))
    return int(glyph.sum()), float(a[glyph].sum())


def clean_borders(img: np.ndarray, box: tuple, borders: dict, fill_bgr: np.ndarray, dpi: float,
                  cfg: dict, acfg: dict) -> tuple:
    """Pale bands: fill with fill_bgr, keeping strokes that cross into them.
    Dark bands: flatten the shadow (keeps text printed on it); fill only unrecoverable / detached parts.
    Returns (cleaned image, filled mask, restored-stroke mask, flattened mask)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    px = lambda mm, odd=False: mm_to_px(mm, dpi, odd)

    band = np.ones((h, w), bool)
    band[y0:y1, x0:x1] = False
    if not band.any():
        return img, band, np.zeros_like(band), np.zeros_like(band), np.zeros_like(band), {
            "inverted_glyphs": 0, "inverted_mm2": 0.0, "inverted_found": False, "dark_untouched_pct": 0.0}

    # Strokes on the whole page, then the ones that cross the content-box edge
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sk = px(acfg["stroke_width_max_mm"], odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    ink = (bh > acfg["stroke_ink"]).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    p = px(cfg["protect_mm"])
    inner_ring = np.zeros((h, w), bool)
    inner_ring[max(0, y0 - 1):min(h, y1 + 1), max(0, x0 - 1):min(w, x1 + 1)] = True
    inner_ring[y0 + p:max(y0 + p, y1 - p), x0 + p:max(x0 + p, x1 - p)] = False
    inner_ring &= ~band
    crossing = np.intersect1d(np.unique(labels[inner_ring & (ink > 0)]), np.unique(labels[band & (ink > 0)]))
    crossing = crossing[crossing > 0]
    max_area = cfg["restore_max_stroke_mm2"] * (dpi / 25.4) ** 2
    crossing = crossing[stats[crossing, cv2.CC_STAT_AREA] <= max_area]        # pen strokes only, not shadow texture
    if crossing.size:                                                         # ... and thin: widest point of each blob
        width = cv2.distanceTransform(ink, cv2.DIST_L2, 3) * 2
        widest = np.zeros(labels.max() + 1, np.float32)
        np.maximum.at(widest, labels[ink > 0], width[ink > 0])
        crossing = crossing[widest[crossing] <= px(cfg["restore_max_stroke_width_mm"])]
    content = (~band).astype(np.uint8)
    reach = cv2.distanceTransform(1 - content, cv2.DIST_L2, 3) <= px(cfg["restore_reach_mm"])
    local_bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    paper_level = float(np.median(g[~band])) if (~band).any() else 255.0
    on_paper = local_bg >= cfg["restore_min_background_rel"] * paper_level
    restored = np.isin(labels, crossing) & band & reach & on_paper
    if restored.any():
        rk = px(cfg["restore_pad_mm"], odd=True)
        restored = cv2.dilate(restored.astype(np.uint8), np.ones((rk, rk), np.uint8)).astype(bool) & band

    # Dark bands: divide out the shadow (keeps text printed on it). The part beyond the spine ("outer") is
    # flattened too, never erased: it may be this page continuing across the spine. Its ink is reported.
    work = img.astype(np.float32)
    outer, inner = dark_band_split(g, borders, dpi, cfg["crease_smooth_mm"], cfg["shadow_edge_rel"])
    outer &= band
    inner = (inner | outer) & band
    beyond_ink = outer & (ink > 0)
    flattened = np.zeros((h, w), bool)
    # White-on-dark marks inside the dark band (e.g. a black title banner near the edge) are reported for review.
    # They are safe either way: flattening divides by the local BRIGHT level (gain ~1 on a banner), and parts too
    # dark to flatten are left as original (never painted) unless fill_unrecoverable_dark is set.
    n_glyphs, glyph_mm2 = inverted_glyphs(g, inner, dpi, cfg, acfg, paper_level)
    dark_untouched = np.zeros((h, w), bool)
    if inner.any():
        sm = max(1.0, cfg["shadow_smooth_mm"] * dpi / 25.4)
        bk = px(cfg["shadow_bg_mm"], odd=True)
        bg = cv2.morphologyEx(cv2.GaussianBlur(g, (0, 0), sm), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (bk, bk)))
        bg = cv2.GaussianBlur(bg, (0, 0), bk / 2).astype(np.float32)
        recoverable = (bg >= paper_level / cfg["shadow_max_gain"]).astype(np.uint8)
        mk = px(cfg["shadow_mask_smooth_mm"], odd=True)
        recoverable = cv2.morphologyEx(cv2.morphologyEx(recoverable, cv2.MORPH_OPEN, np.ones((mk, mk), np.uint8)),
                                       cv2.MORPH_CLOSE, np.ones((mk, mk), np.uint8)).astype(bool)
        flattened = inner & recoverable
        if not cfg["fill_unrecoverable_dark"]:
            dark_untouched = inner & ~recoverable              # too dark to flatten: keep the original pixels
        # Flatten lightness only (Lab L x gain); colour channels untouched so JPEG chroma noise is not amplified
        gain = np.where(flattened, paper_level / np.maximum(bg, 1.0), 1.0)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        paper_l = float(cv2.cvtColor(fill_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0, 0])
        lab[..., 0] = np.minimum(lab[..., 0] * gain, paper_l)
        flat = cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        # Amplified scanner grain: where the blurred result is close to paper there is no stroke -> paper colour
        fg = cv2.GaussianBlur(cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_BGR2GRAY), (0, 0),
                              max(0.5, cfg["shadow_grain_blur_mm"] * dpi / 25.4))
        grain = flattened & (fg >= cfg["shadow_grain_rel"] * paper_level)
        flat[grain] = fill_bgr
        work = np.where(flattened[..., None], flat, work)

    if cfg["shadow_lift"] and flattened.any():                 # lighten the flattened shadow grain, keep text
        region = dark_band_mask(h, w, borders) & band
        work = shadow_lift(np.clip(work + 0.5, 0, 255).astype(np.uint8), region, fill_bgr, dpi,
                           cfg["shadow_lift"], cfg["shadow_lift_feather_mm"]).astype(np.float32)

    fill = band & ~restored & ~flattened & ~dark_untouched
    # Feather: alpha rises from 0 at the kept region to 1 at feather_mm inside the filled area
    keep = (~fill).astype(np.uint8)
    dist = cv2.distanceTransform(1 - keep, cv2.DIST_L2, 3)
    alpha = np.clip(dist / max(1.0, px(cfg["feather_mm"])), 0, 1)[..., None]
    out = work * (1 - alpha) + fill_bgr[None, None, :] * alpha
    info = {"inverted_glyphs": n_glyphs, "inverted_mm2": glyph_mm2,
            "inverted_found": n_glyphs >= cfg["inverted_min_glyphs"],
            "dark_untouched_pct": round(float(dark_untouched.mean() * 100), 2)}
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), fill, restored, flattened, beyond_ink, info


def step4_borders(rec: dict) -> None:
    if rec["metrics"].get(STEP4):                              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP4]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP4]
    cfg, review = CONFIG["borders"], CONFIG["review"]["borders"]
    img = rec["images"]["oriented"]
    s3 = rec["metrics"][STEP3]
    if not rec["needs"]["border_cleanup"] or rec["needs"]["skip_page"]:
        rec["images"]["borders_clean"] = img                   # unchanged (no copy)
        rec["metrics"][STEP4] = {"applied": False, "filled_pct": 0.0, "restored_mm2": 0.0}
        add_action(rec, STEP4, "no border to clean")
        return

    t0 = time.perf_counter()
    fill_bgr = (np.array([255, 255, 255], np.float32) if cfg["fill"] == "white"
                else paper_bgr(rec["metrics"][STEP2]["paper_lab"]))
    out, filled, restored, flattened, beyond_ink, inv = clean_borders(img, s3["content_box"], s3["borders"], fill_bgr, rec["dpi"],
                                                     cfg, CONFIG["assess"])
    rec["images"]["borders_clean"] = out

    h, w = out.shape[:2]
    mm2 = (25.4 / rec["dpi"]) ** 2
    filled_pct = float(filled.mean() * 100)
    restored_mm2 = float(restored.sum() * mm2)
    beyond_mm2 = float(beyond_ink.sum() * mm2)
    n_glyphs, glyph_mm2, inv_found = inv["inverted_glyphs"], inv["inverted_mm2"], inv["inverted_found"]
    residual, _ = detect_borders(out, CONFIG["assess"], rec["dpi"])     # verify: no dark strip should be left
    remaining = {s: b["px"] for s, b in residual.items() if b["px"] > 0 and "dark" in (b["kind"] or "")}
    pale_left = {s: b["px"] for s, b in residual.items() if b["kind"] == "pale"}
    rec["metrics"][STEP4] = {
        "applied": True,
        "fill_bgr": [round(float(v)) for v in fill_bgr],
        "filled_pct": round(filled_pct, 2),
        "shadow_flattened_pct": round(float(flattened.mean() * 100), 2),
        "restored_mm2": round(restored_mm2, 1),
        "beyond_spine_ink_mm2": round(beyond_mm2, 1),
        "inverted_glyphs": n_glyphs, "dark_kept_original_pct": inv["dark_untouched_pct"],
        "sides": {s: b["px"] for s, b in s3["borders"].items() if b["px"]},
        "remaining_after": remaining,
        "pale_after": pale_left,              # info only: filled bands can differ slightly from the paper reference
        "time_ms": round((time.perf_counter() - t0) * 1000),
    }
    sides = ", ".join(f"{s} {v}px" for s, v in rec["metrics"][STEP4]["sides"].items())
    add_action(rec, STEP4, f"filled {filled_pct:.1f}% of page with paper colour ({sides}); "
                           f"flattened {flattened.mean() * 100:.1f}% (spine shadow, text kept); "
                           f"kept {restored_mm2:.1f} mm² of strokes crossing the edge")
    if restored_mm2 > review["text_at_border_mm2"]:
        add_flag(rec, STEP4, "text_at_border", f"{restored_mm2:.1f} mm² of strokes cross into the border band and were kept: "
                                               "check the band did not cut content")
    if inv_found:
        add_flag(rec, STEP4, "inverted_content_in_border",
                 f"{n_glyphs} white-on-dark glyph-like marks ({glyph_mm2:.0f} mm²) inside a dark border band: "
                 "check them (flattened band keeps them; too-dark parts kept as original)")
    if beyond_mm2 > review["beyond_spine_ink_mm2"]:
        add_flag(rec, STEP4, "content_beyond_spine",
                 f"{beyond_mm2:.0f} mm² of ink beyond the spine shadow was kept: neighbour-page sliver or this page "
                 "continuing across the spine? (not erased)")
    if filled_pct > review["large_fill_pct"]:
        add_flag(rec, STEP4, "large_fill", f"{filled_pct:.1f}% of the page was filled")
    if remaining:
        add_flag(rec, STEP4, "border_remaining", "dark border still detected after cleanup: " +
                 ", ".join(f"{s} {v}px" for s, v in remaining.items()))


# ====================================================================================================
# Step 5: deskew
# ====================================================================================================
STEP5 = "step5_deskew"


def deskew_masks(img: np.ndarray, box: tuple, dpi: float, acfg: dict, work_dpi: float) -> tuple:
    """(text strokes without ruled lines, all strokes) of the content box at work_dpi."""
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    f = work_dpi / dpi
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    px = lambda mm, odd=False: mm_to_px(mm, work_dpi, odd)
    k = px(acfg["stroke_width_max_mm"], odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ink = (bh > acfg["stroke_ink"]).astype(np.uint8)
    gap, run_len = px(acfg["rule_line_gap_mm"], odd=True), px(acfg["rule_line_min_mm"], odd=True)
    h_lines = cv2.morphologyEx(cv2.dilate(ink, np.ones((1, gap), np.uint8)), cv2.MORPH_OPEN, np.ones((1, run_len), np.uint8))
    v_lines = cv2.morphologyEx(cv2.dilate(ink, np.ones((gap, 1), np.uint8)), cv2.MORPH_OPEN, np.ones((run_len, 1), np.uint8))
    lines = cv2.dilate(h_lines | v_lines, np.ones((3, 3), np.uint8))
    return ink & (1 - lines), ink


def _profile_score(mask: np.ndarray, angle: float) -> float:
    """Sharpness of the row profile after rotating by angle. Nearest-neighbour on purpose: interpolation would
    blur every angle except 0° and bias the search towards 'no skew'."""
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    p = rot.sum(axis=1, dtype=np.float64)
    return float(np.sum(np.diff(p) ** 2))


def projection_skew(text_mask: np.ndarray, cfg: dict) -> tuple:
    """Correction angle (deg, cv2 convention: rotate by this to straighten) and peak sharpness vs 0°."""
    if int(text_mask.sum()) < cfg["min_text_px"]:
        return None, None
    m = text_mask.astype(np.uint8)
    lim = cfg["max_angle"]
    coarse = np.arange(-lim, lim + 1e-6, cfg["coarse_step"])
    scores = [_profile_score(m, a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])
    fine = np.arange(best - cfg["coarse_step"], best + cfg["coarse_step"] + 1e-6, cfg["fine_step"])
    fscores = [_profile_score(m, a) for a in fine]
    best = float(fine[int(np.argmax(fscores))])
    base = _profile_score(m, 0.0)
    peak = max(fscores) / base if base > 0 else None
    return best, peak


def line_skew(ink: np.ndarray, work_dpi: float, cfg: dict) -> tuple:
    """Correction angle from long straight segments (form rules / dotted lines); (angle, n_segments, total_mm)."""
    px = lambda mm: mm_to_px(mm, work_dpi)
    segs = cv2.HoughLinesP(ink * 255, 1, np.pi / 1800, px(cfg["line_votes_mm"]),
                           minLineLength=px(cfg["line_min_mm"]), maxLineGap=px(cfg["line_gap_mm"]))
    if segs is None or len(segs) == 0:
        return None, 0, 0.0
    s = np.asarray(segs).reshape(-1, 4).astype(np.float64)     # OpenCV 4: Nx1x4, OpenCV 5: Nx4
    dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    ang = np.degrees(np.arctan2(dy, dx))
    ang = (ang + 90) % 180 - 90                          # direction-free, in (-90, 90]
    keep = np.abs(ang) <= cfg["max_angle"]
    if keep.sum() < cfg["min_lines"]:
        return None, int(keep.sum()), 0.0
    ang, length = ang[keep], np.hypot(dx, dy)[keep]
    order = np.argsort(ang)
    cum = np.cumsum(length[order])
    median = float(ang[order][np.searchsorted(cum, cum[-1] / 2)])  # length-weighted median
    # image y points down: a segment sloping down-right (angle > 0) is straightened by rotating +angle in cv2
    return median, int(keep.sum()), float(length.sum() * 25.4 / work_dpi)


def warp_affine(img: np.ndarray, M: np.ndarray, size: tuple, border) -> np.ndarray:
    """warpAffine on the device's warp backend (OpenCL UMat / CUDA), CPU fallback."""
    backend = PROFILE["backends"]["warp"]
    try:
        if backend == "cuda":
            gm = cv2.cuda_GpuMat(); gm.upload(img)
            return cv2.cuda.warpAffine(gm, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=border).download()
        if backend == "opencl":
            return cv2.warpAffine(cv2.UMat(img), M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=border).get()
    except cv2.error:
        pass
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def rotate_expand(img: np.ndarray, angle: float, border) -> tuple:
    """Rotate about the centre by angle (cv2 convention), canvas grown so nothing is clipped. Returns (img, M)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(w * cos + h * sin)), int(np.ceil(w * sin + h * cos))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return warp_affine(img, M, (nw, nh), border), M


def transform_box(box: tuple, M: np.ndarray, size: tuple) -> tuple:
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], np.float64)
    t = pts @ M[:, :2].T + M[:, 2]
    w, h = size
    return (int(max(0, np.floor(t[:, 0].min()))), int(max(0, np.floor(t[:, 1].min()))),
            int(min(w, np.ceil(t[:, 0].max()))), int(min(h, np.ceil(t[:, 1].max()))))


def estimate_skew(img: np.ndarray, box: tuple, dpi: float, cfg: dict) -> dict:
    """Combine the line and projection estimates by how trustworthy each one is (see CONFIG['deskew'])."""
    text, ink = deskew_masks(img, box, dpi, CONFIG["assess"], cfg["work_dpi"])
    p_angle, peak = projection_skew(text, cfg)
    l_angle, n_lines, line_mm = line_skew(ink, cfg["work_dpi"], cfg)
    lines_strong = l_angle is not None and n_lines >= cfg["lines_strong_min"] and line_mm >= cfg["lines_strong_mm"]
    proj_strong = p_angle is not None and peak is not None and peak >= cfg["projection_strong_peak"]
    close = (l_angle is not None and p_angle is not None and abs(l_angle - p_angle) <= cfg["agree_deg"])

    if lines_strong:
        angle, source = l_angle, "lines"
    elif proj_strong and (l_angle is None or abs(p_angle - l_angle) <= cfg["projection_conflict_deg"]):
        angle, source = p_angle, "projection"
    elif close:
        angle, source = l_angle, "both_weak_agree"
    else:
        angle, source = 0.0, "uncertain" if (l_angle is not None or p_angle is not None) else "none"
    disagree = lines_strong and proj_strong and not close
    suggested = max([abs(a) for a in (l_angle, p_angle) if a is not None], default=0.0)
    return {"angle": round(angle, 2), "source": source, "disagree": disagree,
            "uncertain": source in ("uncertain", "none") and suggested >= cfg["uncertain_flag_deg"],
            "projection_angle": None if p_angle is None else round(p_angle, 2),
            "projection_peak": None if peak is None else round(peak, 3),
            "line_angle": None if l_angle is None else round(l_angle, 2),
            "n_lines": n_lines, "line_length_mm": round(line_mm)}


def step5_deskew(rec: dict) -> None:
    if rec["metrics"].get(STEP5):                              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP5]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP5]
    cfg, review = CONFIG["deskew"], CONFIG["review"]["deskew"]
    img, box = rec["images"]["borders_clean"], rec["metrics"][STEP3]["content_box"]
    t0 = time.perf_counter()
    if rec["needs"]["skip_page"]:
        est = {"angle": 0.0, "source": "blank_not_checked", "disagree": False, "uncertain": False,
               "projection_angle": None, "projection_peak": None, "line_angle": None, "n_lines": 0, "line_length_mm": 0}
    else:
        est = estimate_skew(img, box, rec["dpi"], cfg)

    angle = est["angle"]
    if abs(angle) >= cfg["min_rotate_deg"]:
        border = tuple(float(v) for v in paper_bgr(rec["metrics"][STEP2]["paper_lab"]))
        out, M = rotate_expand(img, angle, border)
        new_box = transform_box(box, M, (out.shape[1], out.shape[0]))
        add_action(rec, STEP5, f"rotated {angle:+.2f}° ({est['source']}); canvas {img.shape[1]}x{img.shape[0]} -> "
                               f"{out.shape[1]}x{out.shape[0]}")
    else:
        out, new_box = img, box
        add_action(rec, STEP5, f"skew {angle:+.2f}° below {cfg['min_rotate_deg']}°: not rotated")
    rec["images"]["deskewed"] = out
    rec["metrics"][STEP5] = {**est, "rotated": out is not img, "content_box": new_box,
                             "orientation": "landscape" if out.shape[1] > out.shape[0] else "portrait",
                             "time_ms": round((time.perf_counter() - t0) * 1000)}

    if est["disagree"]:
        add_flag(rec, STEP5, "deskew_disagree", f"lines say {est['line_angle']:+.2f}°, text rows say "
                                                f"{est['projection_angle']:+.2f}° (both strong): used lines")
    if abs(angle) > review["large_skew_deg"]:
        add_flag(rec, STEP5, "large_skew", f"rotated {angle:+.2f}° ({est['source']}): slanted handwriting or real skew?")
    if est["uncertain"]:
        add_flag(rec, STEP5, "deskew_uncertain", f"no trustworthy estimate (lines {est['line_angle']}, text "
                                                 f"{est['projection_angle']} peak {est['projection_peak']}): not rotated")


# ====================================================================================================
# Step 6: illumination & paper colour
# ====================================================================================================
STEP6 = "step6_illumination"


def estimate_paper(img: np.ndarray, dpi: float, cfg: dict) -> np.ndarray:
    """Per-channel paper colour at every pixel (float32, full size), and the fraction treated as dark objects."""
    h, w = img.shape[:2]
    f = cfg["bg_dpi"] / dpi
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    k = mm_to_px(cfg["bg_window_mm"], cfg["bg_dpi"], odd=True)
    s = mm_to_px(cfg["bg_smooth_mm"], cfg["bg_dpi"], odd=True, min_px=3)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.medianBlur(bg, s)
    # Dark OBJECTS wider than the window (photo, thumbprint, seal, black banner) survive the closing and would be
    # taken for dark paper and brightened. Where the estimate is far darker than the page's paper, fill it in from
    # the surrounding paper instead, so the object keeps its tones.
    bg_g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    paper_ref = float(np.percentile(bg_g, 90))
    obj = (bg_g < cfg["dark_object_rel"] * paper_ref).astype(np.uint8)
    obj_frac = float(obj.mean())
    if obj.any() and obj_frac < 0.9:
        obj = cv2.dilate(obj, np.ones((3, 3), np.uint8))
        bg = cv2.inpaint(bg, obj, max(3, k // 2), cv2.INPAINT_TELEA)
    bg = cv2.GaussianBlur(bg, (0, 0), s / 2)
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32), obj_frac


def normalize_illumination(img: np.ndarray, dpi: float, cfg: dict) -> tuple:
    """Divide out the paper colour (per channel), soft white point. Returns (image, gain-capped frac, object frac)."""
    paper, obj_frac = estimate_paper(img, dpi, cfg)
    raw_gain = cfg["target"] / np.maximum(paper, 1.0)
    gain = np.clip(raw_gain, cfg["min_gain"], cfg["max_gain"])
    capped = float(np.mean(raw_gain.max(axis=2) > cfg["max_gain"]))
    out = img.astype(np.float32) * gain
    # Soft white point on luminance (not max channel: blue ink has a bright blue channel but is dark):
    # 0 at snap_start, 1 at 255, applied to all channels (keeps hue of ink)
    light = (0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2])[..., None]
    wgt = np.clip((light - cfg["snap_start"]) / max(1.0, 255.0 - cfg["snap_start"]), 0, 1)
    out = out * (1 - wgt) + 255.0 * wgt
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), capped, obj_frac


def paper_ink_stats(img: np.ndarray, box: tuple, dpi: float, acfg: dict) -> dict:
    """Paper lightness/tint/spread and stroke count/colours inside the content box (for before/after checks)."""
    x0, y0, x1, y1 = box
    im = img[y0:y1, x0:x1]
    f = acfg["reference_dpi"] / dpi
    if abs(f - 1) > 0.05:
        im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        dpi = acfg["reference_dpi"]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    k = mm_to_px(acfg["stroke_width_max_mm"], dpi, odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ink = bh > acfg["stroke_ink"]
    pk = mm_to_px(acfg["paper_pad_mm"], dpi, odd=True)
    paper = ~cv2.dilate((bh > acfg["stroke_weak"]).astype(np.uint8), np.ones((pk, pk), np.uint8)).astype(bool)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    p_lab = np.median(lab[paper], axis=0) if paper.any() else np.array([255, 128, 128])
    q = 4
    sg = cv2.resize(g, (max(1, g.shape[1] // q), max(1, g.shape[0] // q)), interpolation=cv2.INTER_AREA)
    ck = mm_to_px(acfg["illum_close_mm"], dpi / q, odd=True)
    mk = mm_to_px(acfg["illum_median_mm"], dpi / q, odd=True, min_px=3)
    bg = cv2.medianBlur(cv2.morphologyEx(sg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))), mk)
    lo, hi = acfg["illum_percentiles"]
    chroma = im.max(axis=2).astype(np.int16) - im.min(axis=2)
    hue = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 0]
    colored = ink & (chroma >= acfg["ink_chroma_min"])
    n_ink = max(1, int(ink.sum()))

    def share(lo_hi):
        lo_h, hi_h = lo_hi
        band = (hue >= lo_h) & (hue < hi_h) if lo_h < hi_h else (hue >= lo_h) | (hue < hi_h)
        return round(float((colored & band).sum()) / n_ink * 100, 1)

    return {
        "paper_L": round(float(p_lab[0]) * 100 / 255, 1),
        "paper_tint": round(float(np.hypot(p_lab[1] - 128.0, p_lab[2] - 128.0)), 1),
        "illumination_spread": round(float(np.percentile(bg, hi) - np.percentile(bg, lo)), 1),
        "ink_px": int(ink.sum()),
        "ink_level": round(float(np.median(g[ink]))) if ink.any() else None,
        "ink_colors": {c: share(acfg["ink_hues"][c]) for c in ("blue", "purple", "red")},
    }


def step6_illumination(rec: dict) -> None:
    if rec["metrics"].get(STEP6):                              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP6]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP6]
    cfg, review = CONFIG["illumination"], CONFIG["review"]["illumination"]
    img, box = rec["images"]["deskewed"], rec["metrics"][STEP5]["content_box"]
    needed = bool(rec["needs"]["illumination"] or rec["needs"]["tint_removal"])
    if rec["needs"]["skip_page"] or not (cfg["apply_to_all_pages"] or needed):
        rec["images"]["illum_norm"] = img
        rec["metrics"][STEP6] = {"applied": False, "needed": needed}
        add_action(rec, STEP6, "not applied")
        return

    t0 = time.perf_counter()
    out, capped, obj_frac = normalize_illumination(img, rec["dpi"], cfg)
    before = paper_ink_stats(img, box, rec["dpi"], CONFIG["assess"])
    after = paper_ink_stats(out, box, rec["dpi"], CONFIG["assess"])
    rec["images"]["illum_norm"] = out
    ink_change = (after["ink_px"] - before["ink_px"]) / max(1, before["ink_px"]) * 100
    colour_loss = max(before["ink_colors"][c] - after["ink_colors"][c] for c in ("blue", "purple", "red"))
    rec["metrics"][STEP6] = {"applied": True, "needed": needed, "before": before, "after": after,
                             "ink_change_pct": round(ink_change, 1), "colour_loss_pts": round(colour_loss, 1),
                             "gain_capped_pct": round(capped * 100, 2), "dark_object_pct": round(obj_frac * 100, 2),
                             "time_ms": round((time.perf_counter() - t0) * 1000)}
    add_action(rec, STEP6, f"paper L {before['paper_L']} -> {after['paper_L']}, tint {before['paper_tint']} -> "
                           f"{after['paper_tint']}, spread {before['illumination_spread']} -> "
                           f"{after['illumination_spread']}" + ("" if needed else " (not needed; consistency)"))

    if -ink_change > review["ink_loss_pct"]:
        add_flag(rec, STEP6, "ink_loss", f"stroke pixels {ink_change:+.1f}% after normalization")
    if colour_loss > review["colour_loss_pts"]:
        add_flag(rec, STEP6, "colour_loss", f"ink colour share dropped by {colour_loss:.1f} points "
                                            f"({before['ink_colors']} -> {after['ink_colors']})")
    if after["paper_tint"] >= review["tint_remaining"]:
        add_flag(rec, STEP6, "tint_remaining", f"paper tint still {after['paper_tint']}")
    if after["illumination_spread"] >= review["illumination_remaining"]:
        add_flag(rec, STEP6, "illumination_remaining", f"paper-level spread still {after['illumination_spread']}")
    if obj_frac * 100 > review["dark_object_pct"]:
        add_flag(rec, STEP6, "dark_object_kept", f"{obj_frac * 100:.1f}% of the page is a dark object (photo, thumbprint, "
                                                 "seal, banner): its tones were kept, not brightened")
    if capped * 100 > review["gain_capped_pct"]:
        add_flag(rec, STEP6, "gain_capped", f"{capped * 100:.1f}% of the page needed more than "
                                            f"{cfg['max_gain']}x gain (large dark area or very dark paper): left darker")


# ====================================================================================================
# Step 7: noise removal
# ====================================================================================================
STEP7 = "step7_denoise"


def stroke_and_faint_masks(g: np.ndarray, dpi: float, acfg: dict) -> tuple:
    """Same definitions as Step 2: (ink, texture dots, isolated specks, faint stroke-like marks) on a grey image."""
    px = lambda mm, odd=False: mm_to_px(mm, dpi, odd)
    sk = px(acfg["stroke_width_max_mm"], odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    weak, ink, strong = bh > acfg["stroke_weak"], bh > acfg["stroke_ink"], bh > acfg["stroke_strong"]
    wk = weak.astype(np.uint8)
    gap, run_len, pad = px(acfg["rule_line_gap_mm"], odd=True), px(acfg["rule_line_min_mm"], odd=True), px(acfg["rule_line_pad_mm"], odd=True)
    h_lines = cv2.morphologyEx(cv2.dilate(wk, np.ones((1, gap), np.uint8)), cv2.MORPH_OPEN, np.ones((1, run_len), np.uint8))
    v_lines = cv2.morphologyEx(cv2.dilate(wk, np.ones((gap, 1), np.uint8)), cv2.MORPH_OPEN, np.ones((run_len, 1), np.uint8))
    lines = cv2.dilate(h_lines | v_lines, np.ones((pad, pad), np.uint8)).astype(bool)
    nk = px(acfg["near_ink_mm"], odd=True)
    near_strong = cv2.dilate(strong.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nk, nk))).astype(bool)
    faint = (weak & ~near_strong & ~lines).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(faint, connectivity=8)
    fa, fw, fh = st[:, cv2.CC_STAT_AREA], st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    mm2_px = (dpi / 25.4) ** 2
    stroke_like = ((fa >= acfg["faint_stroke_min_area_mm2"] * mm2_px) & (np.minimum(fw, fh) >= px(acfg["faint_stroke_min_size_mm"]))
                   & (fa / np.maximum(1, fw * fh) < acfg["faint_stroke_max_fill"]))
    stroke_like[0] = True                                      # background label
    texture = ~stroke_like[lab]
    faint_strokes = stroke_like[lab] & (lab > 0)                # bleed-through candidates (Step 9)
    # isolated specks among ink
    _, lab2, st2, cen = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    win = px(acfg["speck_isolation_mm"], odd=True)
    box = cv2.boxFilter(ink.astype(np.float32), -1, (win, win), normalize=False)
    cx = np.clip(np.rint(cen[:, 0]).astype(int), 0, g.shape[1] - 1)
    cy = np.clip(np.rint(cen[:, 1]).astype(int), 0, g.shape[0] - 1)
    a2 = st2[:, cv2.CC_STAT_AREA]
    is_speck = (a2 <= acfg["speck_max_area_mm2"] * mm2_px) & (box[cy, cx] <= a2 + 0.5)
    is_speck[0] = False
    specks = is_speck[lab2]
    return ink, texture, specks, faint_strokes


def stroke_pixels(img: np.ndarray, dpi: float, min_mm2: float, acfg: dict) -> int:
    """Ink pixels in connected strokes of at least min_mm2 (grain and specks excluded)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sk = mm_to_px(acfg["stroke_width_max_mm"], dpi, odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    ink = (bh > acfg["stroke_ink"]).astype(np.uint8)
    _, _, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    a = st[1:, cv2.CC_STAT_AREA]
    return int(a[a >= min_mm2 * (dpi / 25.4) ** 2].sum())


def nlm_luminance(L: np.ndarray, h: float, tw: int, sw: int) -> np.ndarray:
    backend = PROFILE["backends"]["denoise"]
    try:
        if backend == "cuda":
            gm = cv2.cuda_GpuMat(); gm.upload(L)
            return cv2.cuda.fastNlMeansDenoising(gm, h, search_window=sw, block_size=tw).download()
        if backend == "opencl":
            return cv2.fastNlMeansDenoising(cv2.UMat(L), None, h, tw, sw).get()
    except cv2.error:
        pass
    return cv2.fastNlMeansDenoising(L, None, h, tw, sw)


def denoise_page(img: np.ndarray, dpi: float, plan: dict, cfg: dict, acfg: dict) -> tuple:
    """Apply the planned fixes. Returns (image, pixels changed by texture cleanup, by speck removal)."""
    px = lambda mm, odd=False: mm_to_px(mm, dpi, odd)
    out = img
    if plan["nlm"] != "none":
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        tw, sw = px(cfg["nlm_template_mm"], odd=True), px(cfg["nlm_search_mm"], odd=True)
        L = nlm_luminance(L, cfg["nlm_h"][plan["nlm"]], tw, sw)
        if plan["chroma"]:                                       # colour speckle on PAPER only: ink keeps its colour
            ck = px(cfg["chroma_median_mm"], odd=True)
            sk = px(acfg["stroke_width_max_mm"], odd=True)
            bh_l = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
            ink_px = cv2.dilate((bh_l > acfg["stroke_weak"]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            a = np.where(ink_px, a, cv2.medianBlur(a, ck))
            b = np.where(ink_px, b, cv2.medianBlur(b, ck))
        out = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
    n_tex = n_spk = 0
    if plan["texture"] or plan["specks"]:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        _, texture, specks, _ = stroke_and_faint_masks(g, dpi, acfg)
        # Only NEUTRAL (grey) marks are ever cleared: toner grain and scanner specks are grey; stamps, seals and
        # coloured ink dots are not. Texture clearing is further limited to faint marks (black-hat <= stroke_ink).
        neutral = (out.max(axis=2).astype(np.int16) - out.min(axis=2)) < acfg["ink_chroma_min"]
        sk = px(acfg["stroke_width_max_mm"], odd=True)
        faint = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk))) <= acfg["stroke_ink"]
        texture &= neutral & faint
        specks &= neutral
        clear = np.zeros(g.shape, bool)
        if plan["texture"]:
            clear |= texture; n_tex = int(texture.sum())
        if plan["specks"]:
            clear |= specks; n_spk = int(specks.sum())
        if clear.any():
            out = out.copy()
            out[clear] = cfg["texture_target"]
    return out, n_tex, n_spk


def step7_denoise(rec: dict) -> None:
    if rec["metrics"].get(STEP7):                              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP7]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP7]
    cfg, review, need = CONFIG["denoise"], CONFIG["review"]["denoise"], CONFIG["assess"]["needs"]
    img = rec["images"]["illum_norm"]
    if rec["needs"]["skip_page"]:
        rec["images"]["denoised"] = img
        rec["metrics"][STEP7] = {"applied": False}
        return
    t0 = time.perf_counter()
    before = assess_page(img, CONFIG["assess"], rec["dpi"])
    if before["noise_sigma"] >= need["denoise_strong_sigma"]:
        nlm = "strong"
    elif before["noise_sigma"] >= need["denoise_mild_sigma"] or before["texture_pct"] >= need["denoise_texture_pct"]:
        nlm = "mild"
    else:
        nlm = "none"
    is_colour = not rec["metrics"][STEP1]["greyscale_scan"]
    plan = {"nlm": nlm, "chroma": nlm != "none" and is_colour,
            "texture": before["texture_pct"] >= cfg["texture_cleanup_pct"],
            "specks": before["isolated_specks_per_mp"] >= need["speck_removal_per_mp"]}
    if not any([plan["nlm"] != "none", plan["texture"], plan["specks"]]):
        rec["images"]["denoised"] = img
        rec["metrics"][STEP7] = {"applied": False, "plan": plan, "before": {k: before[k] for k in
                                 ("noise_sigma", "texture_pct", "isolated_specks_per_mp", "ink_pct")}}
        add_action(rec, STEP7, "no noise to remove")
        return

    out, n_tex, n_spk = denoise_page(img, rec["dpi"], plan, cfg, CONFIG["assess"])
    after = assess_page(out, CONFIG["assess"], rec["dpi"])
    keys = ("noise_sigma", "texture_pct", "isolated_specks_per_mp", "ink_pct")
    s_before = stroke_pixels(img, rec["dpi"], cfg["stroke_min_mm2"], CONFIG["assess"])
    s_after = stroke_pixels(out, rec["dpi"], cfg["stroke_min_mm2"], CONFIG["assess"])
    ink_change = (s_after - s_before) / max(1, s_before) * 100                 # real strokes only
    rec["images"]["denoised"] = out
    rec["metrics"][STEP7] = {"applied": True, "plan": plan, "before": {k: before[k] for k in keys},
                             "after": {k: after[k] for k in keys}, "ink_change_pct": round(ink_change, 1),
                             "texture_px_cleared": n_tex, "speck_px_cleared": n_spk,
                             "time_ms": round((time.perf_counter() - t0) * 1000)}
    done = [f"NLM {nlm}"] if nlm != "none" else []
    done += ["chroma median"] if plan["chroma"] else []
    done += [f"texture dots cleared ({n_tex} px)"] if plan["texture"] else []
    done += [f"isolated specks cleared ({n_spk} px)"] if plan["specks"] else []
    add_action(rec, STEP7, "; ".join(done) + f" | noise {before['noise_sigma']} -> {after['noise_sigma']}, "
                           f"texture {before['texture_pct']} -> {after['texture_pct']}%, "
                           f"specks {before['isolated_specks_per_mp']} -> {after['isolated_specks_per_mp']}/MP")
    if -ink_change > review["ink_loss_pct"]:
        add_flag(rec, STEP7, "ink_loss", f"real-stroke pixels {ink_change:+.1f}% (all ink {before['ink_pct']}% -> {after['ink_pct']}%)")
    if after["noise_sigma"] >= review["noise_remaining_sigma"]:
        add_flag(rec, STEP7, "noise_remaining", f"noise sigma still {after['noise_sigma']}")
    if after["isolated_specks_per_mp"] >= review["specks_remaining_per_mp"]:
        add_flag(rec, STEP7, "specks_remaining", f"{after['isolated_specks_per_mp']} isolated specks/MP remain")


# ====================================================================================================
# Step 8: sharpen
# ====================================================================================================
STEP8 = "step8_sharpen"


def sharpen_luminance(img: np.ndarray, dpi: float, amount: float, cfg: dict) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    detail = L - cv2.GaussianBlur(L, (0, 0), max(0.5, cfg["sigma_mm"] * dpi / 25.4))
    t = cfg["detail_threshold"]
    weight = np.clip((np.abs(detail) - t) / max(1.0, t), 0, 1)           # soft threshold: no boost below t
    sharp = L + amount * detail * weight
    k = np.ones((3, 3), np.uint8)
    lo = cv2.erode(lab[..., 0], k).astype(np.float32) - cfg["halo_overshoot"]
    hi = cv2.dilate(lab[..., 0], k).astype(np.float32) + cfg["halo_overshoot"]
    lab[..., 0] = np.clip(np.clip(sharp, lo, hi) + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_amount(sharpness: float, cfg: dict) -> float:
    return float(np.clip(cfg["gain"] * (cfg["target_sharpness"] - sharpness) / cfg["target_sharpness"],
                         cfg["min_amount"], cfg["max_amount"]))


def step8_sharpen(rec: dict) -> None:
    if rec["metrics"].get(STEP8):                              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP8]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP8]
    cfg, review = CONFIG["sharpen"], CONFIG["review"]["sharpen"]
    img = rec["images"]["denoised"]
    if rec["needs"]["skip_page"]:
        rec["images"]["sharpened"] = img
        rec["metrics"][STEP8] = {"applied": False}
        return
    t0 = time.perf_counter()
    before = assess_page(img, CONFIG["assess"], rec["dpi"])
    if before["sharpness"] >= CONFIG["assess"]["needs"]["sharpen_below"]:
        rec["images"]["sharpened"] = img
        rec["metrics"][STEP8] = {"applied": False, "sharpness": before["sharpness"]}
        add_action(rec, STEP8, f"sharp enough ({before['sharpness']})")
        return

    amount = sharpen_amount(before["sharpness"], cfg)
    out = sharpen_luminance(img, rec["dpi"], amount, cfg)
    after = assess_page(out, CONFIG["assess"], rec["dpi"])
    s_b = stroke_pixels(img, rec["dpi"], CONFIG["denoise"]["stroke_min_mm2"], CONFIG["assess"])
    s_a = stroke_pixels(out, rec["dpi"], CONFIG["denoise"]["stroke_min_mm2"], CONFIG["assess"])
    stroke_change = (s_a - s_b) / max(1, s_b) * 100
    rec["images"]["sharpened"] = out
    rec["metrics"][STEP8] = {"applied": True, "amount": round(amount, 2),
                             "sharpness": [before["sharpness"], after["sharpness"]],
                             "noise_sigma": [before["noise_sigma"], after["noise_sigma"]],
                             "stroke_change_pct": round(stroke_change, 1),
                             "time_ms": round((time.perf_counter() - t0) * 1000)}
    add_action(rec, STEP8, f"unsharp amount {amount:.2f}: sharpness {before['sharpness']} -> {after['sharpness']}, "
                           f"noise {before['noise_sigma']} -> {after['noise_sigma']}")
    if after["sharpness"] < review["still_blurry_below"]:
        add_flag(rec, STEP8, "still_blurry", f"sharpness {after['sharpness']} after sharpening")
    if after["noise_sigma"] - before["noise_sigma"] > review["noise_rise_sigma"]:
        add_flag(rec, STEP8, "noise_amplified", f"noise sigma {before['noise_sigma']} -> {after['noise_sigma']}")
    if abs(stroke_change) > review["stroke_change_pct"]:
        add_flag(rec, STEP8, "ink_change", f"real-stroke pixels {stroke_change:+.1f}%")


# ====================================================================================================
# Step 9: bleed-through
# ====================================================================================================
STEP9 = "step9_bleed"


def ink_components(img: np.ndarray, dpi: float, cfg: dict, acfg: dict) -> tuple:
    """Ink components with their area, mean lightness, mean chroma and mean hue."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sk = mm_to_px(acfg["stroke_width_max_mm"], dpi, odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    ink = (bh > acfg["stroke_ink"]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    idx = lab.ravel()
    m = idx > 0
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].ravel().astype(np.float64)
    chroma = (img.max(axis=2).astype(np.int16) - img.min(axis=2)).ravel().astype(np.float64)
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0].ravel().astype(np.float64)
    area = np.bincount(idx[m], minlength=n).astype(np.float64)
    mean = lambda v: np.bincount(idx[m], weights=v[m], minlength=n) / np.maximum(area, 1)
    comps = {"area": area, "L": mean(L), "chroma": mean(chroma), "hue": mean(hue)}
    big = area >= cfg["min_component_mm2"] * (dpi / 25.4) ** 2
    big[0] = False
    return lab, comps, big


def otsu_split(values: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted Otsu split of 1-D values: threshold, separability eta, class means, light share."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw, cwv = np.cumsum(w), np.cumsum(w * v)
    tot, totv = cw[-1], cwv[-1]
    w0, w1 = cw[:-1], tot - cw[:-1]
    ok = (w0 > 0) & (w1 > 0)
    m0 = np.where(ok, cwv[:-1] / np.maximum(w0, 1e-9), 0)
    m1 = np.where(ok, (totv - cwv[:-1]) / np.maximum(w1, 1e-9), 0)
    between = np.where(ok, w0 * w1 * (m0 - m1) ** 2, -1)
    i = int(np.argmax(between))
    mean = totv / tot
    var = float((w * (v - mean) ** 2).sum() / tot)
    eta = float(between[i] / tot ** 2 / max(var, 1e-9))
    return {"threshold": float((v[i] + v[i + 1]) / 2), "eta": eta, "dark_mean": float(m0[i]),
            "light_mean": float(m1[i]), "light_share": float(w1[i] / tot)}


def detect_bleed(img: np.ndarray, dpi: float, cfg: dict, acfg: dict) -> tuple:
    """Per colour group: is component lightness split into a clean dark (front) and light (show-through) population?
    Returns (labels, light-population component ids, per-group stats)."""
    lab, comps, big = ink_components(img, dpi, cfg, acfg)
    hues = acfg["ink_hues"]
    light_ids, stats = [], {}
    for name in cfg["groups"]:
        lo, hi = hues[name]
        in_band = (comps["hue"] >= lo) & (comps["hue"] < hi) if lo < hi else (comps["hue"] >= lo) | (comps["hue"] < hi)
        sel = big & (comps["chroma"] >= acfg["ink_chroma_min"]) & in_band
        if sel.sum() < cfg["min_components"]:
            stats[name] = {"n": int(sel.sum()), "bleed": False}
            continue
        sp = otsu_split(comps["L"][sel], comps["area"][sel])
        gap = sp["light_mean"] - sp["dark_mean"]
        lo_s, hi_s = cfg["light_share"]
        is_bleed = sp["eta"] >= cfg["min_eta"] and gap >= cfg["min_gap_L"] and lo_s <= sp["light_share"] <= hi_s
        stats[name] = {"n": int(sel.sum()), "eta": round(sp["eta"], 2), "gap": round(gap, 1),
                       "light_share": round(sp["light_share"], 2), "split_L": round(sp["threshold"], 1), "bleed": is_bleed}
        if is_bleed:
            light_ids.extend(np.flatnonzero(sel & (comps["L"] > sp["threshold"])).tolist())
    return lab, np.array(light_ids, dtype=np.int64), stats


def suppress_bleed(img: np.ndarray, groups: dict, dpi: float, cfg: dict, acfg: dict) -> tuple:
    """PIXEL-level: lighten pixels of a bleed colour group that are lighter than its split, away from dark ink of
    that colour and from black/grey ink. (Component-level classification merged bleed with touching front strokes
    and print, and erased them.) Returns (image, lightened mask)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sk = mm_to_px(acfg["stroke_width_max_mm"], dpi, odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    marks = bh > acfg["stroke_weak"]
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    chroma = img.max(axis=2).astype(np.int16) - img.min(axis=2)
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0]
    coloured = chroma >= acfg["ink_chroma_min"]
    pk = mm_to_px(cfg["protect_mm"], dpi, odd=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pk, pk))
    neutral_ink = (bh > acfg["stroke_ink"]) & ~coloured
    protect = cv2.dilate(neutral_ink.astype(np.uint8), kernel).astype(bool)
    mask = np.zeros(g.shape, bool)
    for name, st in groups.items():
        if not st.get("bleed"):
            continue
        lo, hi = acfg["ink_hues"][name]
        band = (hue >= lo) & (hue < hi) if lo < hi else (hue >= lo) | (hue < hi)
        group = marks & coloured & band
        dark = group & (L <= st["split_L"])
        light = group & (L > st["split_L"])
        protect_g = protect | cv2.dilate(dark.astype(np.uint8), kernel).astype(bool)
        mask |= light & ~protect_g
    if not mask.any():
        return img, mask
    alpha = np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.5, cfg["feather_mm"] * dpi / 25.4)) * 1.5,
                    0, 1)[..., None] * cfg["lighten"]
    alpha = np.where(protect[..., None], 0.0, alpha)                         # print stays untouched, also at edges
    out = img.astype(np.float32) * (1 - alpha) + cfg["paper"] * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), mask


def step9_bleed(rec: dict) -> None:
    if rec["metrics"].get(STEP9):                              # re-running the cell: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP9]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP9]
    cfg, review = CONFIG["bleed"], CONFIG["review"]["bleed"]
    img = rec["images"]["sharpened"]
    if rec["needs"]["skip_page"]:
        rec["images"]["bleed_clean"] = img
        rec["metrics"][STEP9] = {"applied": False}
        return
    t0 = time.perf_counter()
    lab, light_ids, stats = detect_bleed(img, rec["dpi"], cfg, CONFIG["assess"])
    step2_said = bool(rec["needs"]["bleed_suppression"])
    if light_ids.size == 0:
        rec["images"]["bleed_clean"] = img
        rec["metrics"][STEP9] = {"applied": False, "groups": stats, "step2_suspected": step2_said}
        add_action(rec, STEP9, "no bleed-through (no colour group splits into two lightness populations)")
        if step2_said:
            add_flag(rec, STEP9, "bleed_not_confirmed", "Step 2 suspected bleed-through but no clean lightness split: "
                                                         "not changed")
        return

    out, mask = suppress_bleed(img, stats, rec["dpi"], cfg, CONFIG["assess"])
    rec["images"]["bleed_mask"] = mask                          # Step 10 must not darken show-through again
    # front strokes = strokes that are not part of the lightened population; compare before/after
    keep = ~cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    f_before = stroke_pixels(np.where(keep[..., None], img, 255).astype(np.uint8), rec["dpi"],
                             CONFIG["denoise"]["stroke_min_mm2"], CONFIG["assess"])
    f_after = stroke_pixels(np.where(keep[..., None], out, 255).astype(np.uint8), rec["dpi"],
                            CONFIG["denoise"]["stroke_min_mm2"], CONFIG["assess"])
    front_change = (f_after - f_before) / max(1, f_before) * 100
    rec["images"]["bleed_clean"] = out
    groups = [g for g, s in stats.items() if s["bleed"]]
    rec["metrics"][STEP9] = {"applied": True, "groups": stats, "step2_suspected": step2_said,
                             "lightened_pct": round(float(mask.mean() * 100), 2),
                             "front_change_pct": round(front_change, 1),
                             "time_ms": round((time.perf_counter() - t0) * 1000)}
    add_action(rec, STEP9, f"lightened light {'/'.join(groups)} ink pixels ({mask.mean() * 100:.2f}% of page) "
                           f"{cfg['lighten']:.0%} toward paper; dark {'/'.join(groups)} and black/grey ink protected")
    add_flag(rec, STEP9, "bleed_suppressed", f"light {'/'.join(groups)} ink treated as show-through and lightened: "
                                             "check no real (lighter) writing was dimmed")
    if abs(front_change) > review["stroke_change_pct"]:
        add_flag(rec, STEP9, "ink_change", f"front stroke pixels {front_change:+.1f}%")


# ====================================================================================================
# Step 10: faint-ink contrast
# ====================================================================================================
STEP10 = "step10_contrast"


def faint_ink_stats(img: np.ndarray, dpi: float, cfg: dict, acfg: dict, return_mask: bool = False):
    """Per-mark faint-ink measure: share of ink area in marks whose core (darkest pixel) is lighter than
    faint_core_min, and the median core of those marks (+ the faint-mark mask if return_mask)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sk = mm_to_px(acfg["stroke_width_max_mm"], dpi, odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    ink = (bh > acfg[cfg.get("_threshold", cfg["measure_threshold"])]).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    idx = lab.ravel()
    m = idx > 0
    gb = cv2.GaussianBlur(g, (3, 3), 0).ravel().astype(np.float64)
    core = np.full(n, 255.0)
    np.minimum.at(core, idx[m], gb[m])
    area = st[:, cv2.CC_STAT_AREA].astype(np.float64)
    big = area >= cfg["faint_min_mark_mm2"] * (dpi / 25.4) ** 2
    big[0] = False
    faint = big & (core > cfg["faint_core_min"])
    total = area[big].sum()
    stats = {"faint_share": round(float(area[faint].sum() / total), 3) if total else 0.0,
             "faint_core": round(float(np.median(core[faint])), 1) if faint.any() else None,
             "dark_core": round(float(np.median(core[big & ~faint])), 1) if (big & ~faint).any() else None}
    return (stats, faint[lab]) if return_mask else stats


def faint_ink_curve(faint_core: float, cfg: dict) -> np.ndarray:
    """Monotone LUT through (0,0), (faint_core, target), (knee, knee), (255,255), smoothed. Below the diagonal:
    ink gets darker, paper (>= knee) is untouched, tone order never changes (nothing can be erased)."""
    knee, target = float(cfg["knee"]), float(cfg["target_faint_core"])
    fc = float(np.clip(faint_core, target + 1, knee - 1))
    target = max(target, knee - cfg["max_faint_slope"] * (knee - fc))      # limit the stretch above the faint core
    x = np.arange(256, dtype=np.float64)
    lut = np.interp(x, [0, fc, knee, 255], [0, target, knee, 255])
    k = int(cfg["smooth"]) | 1
    if k > 1:
        pad = np.pad(lut, (k // 2, k // 2), mode="edge")
        lut = np.convolve(pad, np.ones(k) / k, mode="valid")              # moving average keeps it monotone
    lut[int(knee):] = x[int(knee):]                                        # paper exactly unchanged
    return np.clip(np.round(np.maximum.accumulate(lut)), 0, 255).astype(np.uint8)


def apply_lightness_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def step10_contrast(rec: dict) -> None:
    if rec["metrics"].get(STEP10):                             # re-running: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP10]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP10]
    cfg, review = CONFIG["contrast"], CONFIG["review"]["contrast"]
    img = rec["images"]["bleed_clean"]
    if rec["needs"]["skip_page"]:
        rec["images"]["contrast"] = img
        rec["metrics"][STEP10] = {"applied": False}
        return
    t0 = time.perf_counter()
    st = faint_ink_stats(img, rec["dpi"], cfg, CONFIG["assess"])
    _, faint_mask = faint_ink_stats(img, rec["dpi"], {**cfg, "_threshold": cfg["apply_threshold"]}, CONFIG["assess"],
                                    return_mask=True)
    if st["faint_share"] < cfg["faint_share_min"] or st["faint_core"] is None:
        rec["images"]["contrast"] = img
        rec["metrics"][STEP10] = {"applied": False, **st}
        add_action(rec, STEP10, f"no faint ink to strengthen (faint marks {st['faint_share']:.0%} of ink)")
        return
    lut = faint_ink_curve(st["faint_core"], cfg)
    # only on the faint marks (+ soft edge): everything else keeps its exact pixels
    if rec["images"].get("bleed_mask") is not None:            # show-through page (Step 9): strengthen only GREY
        neutral = (img.max(axis=2).astype(np.int16) - img.min(axis=2)) < CONFIG["assess"]["ink_chroma_min"]
        faint_mask = faint_mask & neutral                       # faint marks (print), never the faint coloured
                                                                # show-through that Step 9 lightened or protected
    pad = 2 * int(cfg["mark_pad_px"]) + 1
    alpha = cv2.GaussianBlur(cv2.dilate(faint_mask.astype(np.uint8), np.ones((pad, pad), np.uint8)).astype(np.float32),
                             (0, 0), 0.7)[..., None]
    out = np.clip(apply_lightness_lut(img, lut).astype(np.float32) * alpha + img.astype(np.float32) * (1 - alpha) + 0.5,
                  0, 255).astype(np.uint8)
    before = assess_page(img, CONFIG["assess"], rec["dpi"])
    after = assess_page(out, CONFIG["assess"], rec["dpi"])
    st_after = faint_ink_stats(out, rec["dpi"], {**cfg, "faint_core_min": -1}, CONFIG["assess"])
    core_after = int(lut[int(round(st["faint_core"]))])
    rec["images"]["contrast"] = out
    rec["metrics"][STEP10] = {"applied": True, **st, "faint_core_after": core_after,
                              "curve": {"faint_core": st["faint_core"], "maps_to": core_after, "knee": cfg["knee"]},
                              "noise_sigma": [before["noise_sigma"], after["noise_sigma"]],
                              "texture_pct": [before["texture_pct"], after["texture_pct"]],
                              "time_ms": round((time.perf_counter() - t0) * 1000)}
    add_action(rec, STEP10, f"faint marks {st['faint_share']:.0%} of ink (core {st['faint_core']:.0f}): monotone curve "
                            f"{st['faint_core']:.0f} -> {core_after}, paper (>= {cfg['knee']}) unchanged")
    if core_after >= review["still_faint_core"]:
        add_flag(rec, STEP10, "still_faint", f"faint ink core {st['faint_core']:.0f} -> {core_after}")
    if (after["noise_sigma"] - before["noise_sigma"] > review["noise_rise_sigma"]
            or after["texture_pct"] - before["texture_pct"] > review["texture_rise_pct"]):
        add_flag(rec, STEP10, "noise_amplified", f"noise {before['noise_sigma']} -> {after['noise_sigma']}, "
                                                 f"texture {before['texture_pct']} -> {after['texture_pct']}%")


# ====================================================================================================
# Step 11: final image, black & white (optional), applied-steps record
# ====================================================================================================
STEP11 = "step11_output"
STAGES = ["original", "oriented", "borders_clean", "deskewed", "illum_norm", "denoised", "sharpened", "bleed_clean", "contrast"]


def sauvola_bw(img: np.ndarray, dpi: float, cfg: dict) -> np.ndarray:
    """Sauvola adaptive threshold (white paper = 255, ink = 0), small isolated blobs removed."""
    g = (img.min(axis=2) if cfg["source"] == "min_channel" else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    w = mm_to_px(cfg["window_mm"], dpi, odd=True)
    mean = cv2.boxFilter(g, -1, (w, w), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (w, w), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1 + cfg["k"] * (std / cfg["R"] - 1))
    ink = (g <= thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] < cfg["min_blob_mm2"] * (dpi / 25.4) ** 2
    small[0] = False
    ink[small[lab]] = 0
    return np.where(ink > 0, 0, 255).astype(np.uint8)


def bw_checks(color: np.ndarray, bw: np.ndarray, dpi: float) -> dict:
    """How much of the real ink is black in the B&W, and how much black is far from any ink."""
    acfg = CONFIG["assess"]
    g = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    sk = mm_to_px(acfg["stroke_width_max_mm"], dpi, odd=True)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    ink = (bh > acfg["stroke_ink"]).astype(np.uint8)
    _, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    real = st[:, cv2.CC_STAT_AREA] >= CONFIG["denoise"]["stroke_min_mm2"] * (dpi / 25.4) ** 2
    real[0] = False
    strokes = real[lab] & (bh > acfg["stroke_strong"])        # cores only: soft anti-aliased edges may go white
    black = bw == 0
    near = cv2.dilate((bh > acfg["stroke_weak"]).astype(np.uint8),
                      np.ones((mm_to_px(0.5, dpi, odd=True),) * 2, np.uint8)).astype(bool)
    return {"bw_ink_recall": round(float(black[strokes].mean()) if strokes.any() else 1.0, 3),
            "bw_noise_pct": round(float((black & ~near).mean() * 100), 3),
            "bw_black_pct": round(float(black.mean() * 100), 2)}


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def applied_steps(rec: dict) -> list:
    """Every step in pipeline order: applied or not, why, the parameters actually used, and the result."""
    m, C = rec["metrics"], CONFIG
    why = lambda step: next((a for s, a in reversed(rec["actions"]) if s == step), "")
    flags = lambda step: [f["code"] for f in rec["flags"] if f["step"] == step]
    out = []

    def add(step, name, applied, params, result):
        out.append({"order": len(out) + 1, "step": name, "applied": bool(applied), "why": why(step),
                    "params": params if applied else {}, "result": result, "flags": flags(step)})

    s1 = m.get(STEP1, {})
    add(STEP1, "render", True, {"method": s1.get("method"), "codec": s1.get("codec"), "dpi_used": s1.get("dpi_used"),
                                "upscale": s1.get("upscale"), **({"fallback_dpi": C["render"]["fallback_dpi"]}
                                                                 if s1.get("method") == "rendered" else {})},
        {k: s1.get(k) for k in ("native_px", "output_px", "orientation", "greyscale_scan", "paper_tint")})
    add(STEP2, "assess (measure only)", False, {}, {"needs": rec.get("needs")})
    s3 = m.get(STEP3, {})
    add(STEP3, "orientation", s3.get("rotation", 0) != 0, {"rotation_deg_cw": s3.get("rotation")},
        {k: s3.get(k) for k in ("decision", "axis_score", "updown_score")})
    s4 = m.get(STEP4, {})
    add(STEP4, "border cleanup", s4.get("applied"), {**C["borders"], "fill_bgr": s4.get("fill_bgr")},
        {k: s4.get(k) for k in ("sides", "filled_pct", "shadow_flattened_pct", "restored_mm2", "beyond_spine_ink_mm2",
                                "dark_kept_original_pct", "inverted_glyphs", "remaining_after")})
    s5 = m.get(STEP5, {})
    add(STEP5, "deskew", s5.get("rotated"), {"angle_deg": s5.get("angle"), "source": s5.get("source"),
                                              "interpolation": "cubic", "canvas": "expanded"},
        {k: s5.get(k) for k in ("line_angle", "n_lines", "projection_angle", "projection_peak")})
    s6 = m.get(STEP6, {})
    add(STEP6, "illumination & paper colour", s6.get("applied"), dict(C["illumination"]),
        {k: s6.get(k) for k in ("before", "after", "gain_capped_pct", "dark_object_pct", "ink_change_pct")})
    s7 = m.get(STEP7, {})
    p7 = s7.get("plan", {})
    add(STEP7, "noise removal", s7.get("applied"),
        {"nlm": p7.get("nlm"), "nlm_h": C["denoise"]["nlm_h"].get(p7.get("nlm")),
         "nlm_template_mm": C["denoise"]["nlm_template_mm"], "nlm_search_mm": C["denoise"]["nlm_search_mm"],
         "chroma_median_mm": C["denoise"]["chroma_median_mm"] if p7.get("chroma") else None,
         "texture_cleanup": p7.get("texture"), "speck_removal": p7.get("specks")},
        {k: s7.get(k) for k in ("before", "after", "ink_change_pct", "texture_px_cleared", "speck_px_cleared")})
    s8 = m.get(STEP8, {})
    add(STEP8, "sharpen", s8.get("applied"), {"amount": s8.get("amount"), **{k: C["sharpen"][k] for k in
                                               ("sigma_mm", "detail_threshold", "halo_overshoot")}},
        {k: s8.get(k) for k in ("sharpness", "noise_sigma", "stroke_change_pct")})
    s9 = m.get(STEP9, {})
    add(STEP9, "bleed-through", s9.get("applied"), {k: C["bleed"][k] for k in ("lighten", "protect_mm", "min_eta",
                                                     "min_gap_L", "light_share")},
        {k: s9.get(k) for k in ("groups", "lightened_pct", "front_change_pct")})
    s10 = m.get(STEP10, {})
    add(STEP10, "faint ink", s10.get("applied"), s10.get("curve") or {},
        {k: s10.get(k) for k in ("faint_share", "faint_core", "faint_core_after", "dark_core", "noise_sigma", "texture_pct")})
    s11 = m.get(STEP11, {})
    bw_on = C["output"]["bw"].get("enabled", False)
    add(STEP11, "final image", True, {"black_and_white": dict(C["output"]["bw"]) if bw_on else "off"},
        {k: s11.get(k) for k in ("size_px", "bw_ink_recall", "bw_noise_pct", "bw_black_pct") if k in s11})
    return out


# ====================================================================================================
# Pipeline order
# ====================================================================================================
PIPELINE = [step2_assess, step3_orient, step4_borders, step5_deskew, step6_illumination, step7_denoise,
            step8_sharpen, step9_bleed, step10_contrast]


def document_dpi(doc) -> float | None:
    """Median DPI estimate of a document from the embedded scan sizes only (no decoding)."""
    est = []
    for i in range(len(doc)):
        scan, _ = find_page_scan(doc[i])
        if scan is not None:
            w, h = scan.get_px_size()
            est.append(min(w, h) / CONFIG["render"]["assumed_paper_width_in"])
    return float(np.median(est)) if est else None


# ====================================================================================================
# Information-preservation self-test (planted content traps)
# ====================================================================================================
def make_trap_page(base: dict) -> tuple:
    """Copy of a real page record with planted content traps. Returns (record, {trap: (x0, y0, x1, y1)})."""
    img = base["images"]["original"].copy()
    dpi = base["dpi"]
    px = lambda mm: mm_to_px(mm, dpi)
    h, w = img.shape[:2]
    rng = np.random.default_rng(7)
    boxes = {}
    # 1. black banner with white text at the top edge
    x0, y0, x1, y1 = px(15), px(2), w - px(15), px(2) + px(9)
    img[y0:y1, x0:x1] = 25
    cv2.putText(img, "MEDICO LEGAL REPORT 2024", (x0 + px(5), y1 - px(2)), cv2.FONT_HERSHEY_SIMPLEX,
                dpi / 200 * 1.6, (245, 245, 245), max(2, px(0.6)), cv2.LINE_AA)
    boxes["white_on_black_banner"] = (x0, y0, x1, y1)
    # 2. dark photo block with detail
    x0, y0 = px(25), h // 2 - px(15)
    x1, y1 = x0 + px(25), y0 + px(30)
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
    photo = (60 + 50 * np.sin(xx / px(3)) * np.cos(yy / px(4)) + rng.normal(0, 12, xx.shape)).clip(0, 255)
    img[y0:y1, x0:x1] = photo[..., None].astype(np.uint8)
    boxes["dark_photo"] = (x0, y0, x1, y1)
    # 3. faint purple stamp: grainy ring of dots
    cx, cy, r = w - px(45), h // 2, px(14)
    stamp = np.zeros((h, w), np.uint8)
    cv2.circle(stamp, (cx, cy), r, 255, px(1.2))
    cv2.putText(stamp, "SEAL", (cx - px(9), cy + px(2)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 200 * 1.0, 255, px(0.5))
    stamp &= (rng.random((h, w)) < 0.55).astype(np.uint8) * 255
    img[stamp > 0] = (200, 150, 200)
    boxes["faint_purple_stamp"] = (cx - r - px(2), cy - r - px(2), cx + r + px(2), cy + r + px(2))
    # 4. faint pencil strokes
    x0, y0 = px(25), h - px(60)
    for k in range(4):
        cv2.line(img, (x0, y0 + k * px(4)), (x0 + px(50), y0 + k * px(4) + px(2)), (215, 215, 215), max(1, px(0.25)))
    boxes["faint_pencil"] = (x0, y0 - px(1), x0 + px(50), y0 + px(16))
    # 5. thin blue line
    x0, y0 = w // 2 - px(30), h - px(35)
    cv2.line(img, (x0, y0), (x0 + px(60), y0), (190, 90, 30), 1)
    boxes["thin_blue_line"] = (x0, y0 - 2, x0 + px(60), y0 + 3)
    # 6. light blue writing only (no dark blue)
    x0, y0 = w // 2 - px(40), h - px(25)
    for k in range(3):
        cv2.putText(img, "light blue note", (x0, y0 + k * px(6)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    dpi / 200 * 1.1, (230, 190, 150), max(1, px(0.35)), cv2.LINE_AA)
    boxes["light_blue_only"] = (x0, y0 - px(5), x0 + px(80), y0 + px(14))

    rec = {**{k: v for k, v in base.items() if k not in ("images", "metrics", "flags", "actions", "needs")},
           "doc": base["doc"] + "_TRAP", "images": {"original": img}, "metrics": {STEP1: dict(base["metrics"][STEP1])},
           "flags": [], "actions": []}
    return rec, boxes


def trap_checks(before: np.ndarray, after: np.ndarray, boxes: dict, rec: dict) -> dict:
    """Map each trap box from the original page into the final page (orientation/deskew can move it)."""
    res = {}
    s5 = rec["metrics"][STEP5]
    if s5.get("rotated"):                                      # undo the deskew so both images share coordinates
        hh, ww = before.shape[:2]
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), s5["angle"], 1.0)
        M[0, 2] += after.shape[1] / 2 - ww / 2
        M[1, 2] += after.shape[0] / 2 - hh / 2
        after = cv2.warpAffine(after, cv2.invertAffineTransform(M), (ww, hh), flags=cv2.INTER_LINEAR,
                               borderValue=(255, 255, 255))

    def region(img, box):
        x0, y0, x1, y1 = box
        return img[y0:y1, x0:x1]
    L = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    chroma = lambda im: (im.max(axis=2).astype(np.int16) - im.min(axis=2)).astype(np.float32)

    b, a = region(before, boxes["white_on_black_banner"]), region(after, boxes["white_on_black_banner"])
    bl, al = L(b), L(a)
    text = bl > 150
    c_b = bl[text].mean() - bl[~text].mean()
    c_a = al[text].mean() - al[~text].mean() if text.any() else 0
    res["white_on_black_banner"] = (c_a >= 0.6 * c_b, f"contrast {c_b:.0f} -> {c_a:.0f}")

    b, a = L(region(before, boxes["dark_photo"])), L(region(after, boxes["dark_photo"]))
    ok = abs(a.mean() - b.mean()) <= 0.15 * b.mean() and a.std() >= 0.8 * b.std()
    res["dark_photo"] = (ok, f"mean {b.mean():.0f} -> {a.mean():.0f}, std {b.std():.0f} -> {a.std():.0f}")

    b, a = region(before, boxes["faint_purple_stamp"]), region(after, boxes["faint_purple_stamp"])
    cb = chroma(b) >= 30
    kept = float((chroma(a)[cb] >= 20).mean()) if cb.any() else 1.0
    res["faint_purple_stamp"] = (kept >= 0.8, f"{kept:.0%} of coloured stamp pixels still coloured")

    b, a = L(region(before, boxes["faint_pencil"])), L(region(after, boxes["faint_pencil"]))
    strokes = b < 230
    vis = float((a[strokes] < 235 * 100 / 100).mean()) if strokes.any() else 1.0
    res["faint_pencil"] = (vis >= 0.8, f"{vis:.0%} of pencil pixels still visible (L < 235)")

    b, a = region(before, boxes["thin_blue_line"]), region(after, boxes["thin_blue_line"])
    line = chroma(b) >= 40
    kept = float(chroma(a)[line].mean() / chroma(b)[line].mean()) if line.any() else 1.0
    res["thin_blue_line"] = (kept >= 0.6, f"chroma kept {kept:.0%}")

    res["light_blue_only"] = (not rec["metrics"][STEP9].get("applied"), "Step 9 applied" if rec["metrics"][STEP9].get("applied")
                              else "not treated as bleed-through")
    return res


# ====================================================================================================
# Public API
# ====================================================================================================
def finalize_page(rec: dict) -> None:
    """Step 11 without writing files: the final colour image (+ optional B&W with its checks)."""
    if rec["metrics"].get(STEP11):                             # re-running: start this step clean
        rec["flags"] = [f for f in rec["flags"] if f["step"] != STEP11]
        rec["actions"] = [a for a in rec["actions"] if a[0] != STEP11]
    out_cfg, review = CONFIG["output"], CONFIG["review"]["output"]
    final = rec["images"]["contrast"]
    rec["images"]["final"] = final
    m = {"size_px": [int(final.shape[1]), int(final.shape[0])]}
    if out_cfg["bw"].get("enabled"):
        bw = sauvola_bw(final, rec["dpi"], out_cfg["bw"])
        rec["images"]["final_bw"] = bw
        if not rec["needs"]["skip_page"]:
            m.update(bw_checks(final, bw, rec["dpi"]))
            if m["bw_ink_recall"] < review["bw_ink_recall"]:
                add_flag(rec, STEP11, "bw_ink_loss", f"only {m['bw_ink_recall']:.0%} of stroke cores are black in the B&W")
            if m["bw_noise_pct"] > review["bw_noise_pct"]:
                add_flag(rec, STEP11, "bw_noise", f"{m['bw_noise_pct']}% of the page is black away from any stroke")
    rec["metrics"][STEP11] = m
    add_action(rec, STEP11, "final colour image ready" + (" + black & white" if out_cfg["bw"].get("enabled") else ""))


def process_page(rec: dict) -> dict:
    """Steps 2-11 on one rendered page record, in pipeline order (nothing is written)."""
    for step in PIPELINE:
        step(rec)
    finalize_page(rec)
    return rec


def page_stem(rec: dict) -> str:
    """{pdf name}_{page number}, zero-padded so files sort in page order."""
    digits = max(CONFIG["output"]["page_digits"], len(str(rec.get("n_pages") or 0)))
    return f"{rec['doc']}_{rec['page_no']:0{digits}d}"


def applied_record(rec: dict, files: dict | None = None) -> dict:
    """What was applied to this page, in pipeline order (the content of {pdf}_{page}_applied.json)."""
    return to_jsonable({
        "document": rec["doc"], "source": rec["doc_path"], "page": rec["page_no"], "pages_in_document": rec["n_pages"],
        "dpi": round(rec["dpi"], 1), "device": PROFILE.get("name"),
        "needs_review": bool(rec["flags"]), "flags": rec["flags"],
        "files": files or {}, "steps": applied_steps(rec),
    })


def save_page(rec: dict, output_dir) -> dict:
    """Write {pdf}_{page}_original / _final / _applied.json (flat in output_dir). Returns the file names."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = page_stem(rec)
    comp = [cv2.IMWRITE_PNG_COMPRESSION, CONFIG["output"]["png_compression"]]
    if rec.get("original_bytes"):                              # exact embedded scan
        orig = out / f"{stem}_original.jpg"
        orig.write_bytes(rec["original_bytes"])
    else:                                                      # page was rendered: lossless PNG
        orig = out / f"{stem}_original.png"
        cv2.imwrite(str(orig), rec["images"]["original"], comp)
    final = out / f"{stem}_final.png"
    cv2.imwrite(str(final), rec["images"]["final"], comp)
    files = {"original": orig.name, "final": final.name}
    if rec["images"].get("final_bw") is not None:
        bw = out / f"{stem}_final_bw.png"
        cv2.imwrite(str(bw), rec["images"]["final_bw"], comp + [cv2.IMWRITE_PNG_BILEVEL, 1])
        files["final_bw"] = bw.name
    applied = out / f"{stem}_applied.json"
    files["applied"] = applied.name
    with open(applied, "w", encoding="utf-8") as fh:
        json.dump(applied_record(rec, files), fh, ensure_ascii=False, indent=1)
    rec["files"] = files
    return files


def _deep_merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            if k not in base:
                raise KeyError(f"Unknown CONFIG key: {k!r}")
            base[k] = v
    return base


@dataclass
class PageResult:
    page: int                       # 1-based page number
    original: np.ndarray            # scan as in the PDF (RGB by default, see color=)
    final: np.ndarray               # cleaned colour page (RGB by default): send this to the model
    applied: dict                   # same content as {pdf}_{page}_applied.json
    files: dict = field(default_factory=dict)   # saved file names (empty when save=False)


class DocumentPreprocessor:
    """Clean every page of a scanned PDF. Settings and device are set up once; process any number of PDFs.

    config  : partial CONFIG overrides, e.g. {"deskew": {"max_angle": 8}} (unknown keys raise KeyError)
    device  : None (auto-detect), "mac", "cuda" or "cpu"
    workers : pages processed in parallel (default from the device profile)

    Note: settings are module-wide (the step functions read CONFIG); use one preprocessor configuration at a time.
    """

    def __init__(self, config: dict | None = None, device: str | None = None, workers: int | None = None):
        merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), copy.deepcopy(config or {}))
        CONFIG.clear()
        CONFIG.update(merged)
        overrides = dict(CONFIG.get("device_overrides") or {})
        if workers:
            overrides["num_workers"] = int(workers)
        set_device(overrides, device)

    @staticmethod
    def device_info() -> None:
        print_profile(PROFILE)

    def iter_pages(self, pdf_path, save: bool = False, output_dir=None, color: str = "rgb") -> Iterator[PageResult]:
        """Yield one PageResult per page, in page order. Memory: only a few pages in flight at a time.
        color: "rgb" (default: matplotlib, PIL, AI models) or "bgr" (OpenCV). Saved files are unaffected."""
        if color not in ("rgb", "bgr"):
            raise ValueError('color must be "rgb" or "bgr"')
        conv = (lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) if color == "rgb" else (lambda im: im)
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(pdf_path)
        out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        in_flight = CONFIG["batch"]["pages_in_flight"] or 2 * PROFILE["num_workers"]
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            n = len(doc)
            doc_dpi = (None if CONFIG["render"]["dpi_override"] or not CONFIG["render"]["dpi_per_document"]
                       else document_dpi(doc))
            for start in range(0, n, in_flight):
                chunk = []
                for i in range(start, min(n, start + in_flight)):      # pdfium is not thread-safe: render here
                    rec = new_page_record(pdf_path.stem, pdf_path, i, doc[i], n)
                    rec["images"]["original"] = render_page(doc[i], rec, CONFIG["render"], CONFIG["review"]["render"],
                                                            PROFILE["max_render_scale"])
                    if doc_dpi:
                        rec["dpi"] = doc_dpi * rec["metrics"][STEP1]["upscale"]
                        rec["metrics"][STEP1]["dpi_used"] = round(rec["dpi"])
                    chunk.append(rec)
                with ThreadPoolExecutor(max_workers=PROFILE["num_workers"]) as pool:
                    list(pool.map(process_page, chunk))
                for rec in chunk:
                    files = save_page(rec, out_dir) if save else {}
                    result = PageResult(page=rec["page_no"], original=conv(rec["images"]["original"]),
                                        final=conv(rec["images"]["final"]), applied=applied_record(rec, files), files=files)
                    rec["images"].clear()                               # free intermediate stages
                    rec.pop("original_bytes", None)
                    yield result
        finally:
            doc.close()

    def process(self, pdf_path, save: bool = False, output_dir=None, return_all: bool = False,
                color: str = "rgb") -> list:
        """All pages in page order: list of final images (default) or list of PageResult (return_all=True).
        Images are RGB by default (show with plt.imshow, send to a model as is); color="bgr" for OpenCV.
        save=True writes {pdf}_{page}_original / _final.png / _applied.json to output_dir (default: current dir)."""
        results = []
        for r in self.iter_pages(pdf_path, save=save, output_dir=output_dir, color=color):
            results.append(r if return_all else r.final)
        return results

    def self_test(self, pdf_path="sample-docs/FIR-4-2013.pdf", page: int = 4, verbose: bool = True) -> bool:
        """Plant content traps on a real page, run the pipeline, check nothing was lost (and determinism)."""
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            i = min(page, len(doc)) - 1
            base = new_page_record(Path(pdf_path).stem, Path(pdf_path), i, doc[i], len(doc))
            base["images"]["original"] = render_page(doc[i], base, CONFIG["render"], CONFIG["review"]["render"],
                                                     PROFILE["max_render_scale"])
        finally:
            doc.close()
        results, same, _ = run_trap_test(base)
        ok = all(v for v, _ in results.values()) and same
        if verbose:
            print(f"Self-test on {page_label(base)} (+ planted traps):")
            for name, (v, msg) in results.items():
                print(f"  {'PASS' if v else 'FAIL'}  {name:<24} {msg}")
            print(f"  {'PASS' if same else 'FAIL'}  {'deterministic':<24} identical output on a second run")
        return ok


def run_trap_test(base: dict) -> tuple:
    """(results, deterministic, trap record) for a rendered base page record."""
    trap, boxes = make_trap_page(base)
    process_page(trap)
    results = trap_checks(trap["images"]["original"], trap["images"]["contrast"], boxes, trap)
    trap2, _ = make_trap_page(base)
    process_page(trap2)
    same = bool(np.array_equal(trap["images"]["contrast"], trap2["images"]["contrast"]))
    return results, same, trap


def preprocess_pdf(pdf_path, save: bool = False, output_dir=None, return_all: bool = False,
                   color: str = "rgb", config: dict | None = None) -> list:
    """One-call shortcut: DocumentPreprocessor(config).process(pdf_path, save, output_dir, return_all, color)."""
    return DocumentPreprocessor(config).process(pdf_path, save=save, output_dir=output_dir, return_all=return_all,
                                                color=color)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Clean the pages of scanned FIR PDFs (local, OpenCV only).")
    ap.add_argument("pdf", nargs="+", help="PDF file(s)")
    ap.add_argument("-o", "--output-dir", default=None, help="where to save (default: current directory)")
    ap.add_argument("--no-save", action="store_true", help="process only, do not write files")
    ap.add_argument("--device", choices=["mac", "cuda", "cpu"], default=None)
    args = ap.parse_args()
    pre = DocumentPreprocessor(device=args.device)
    pre.device_info()
    for p in args.pdf:
        t0 = time.perf_counter()
        n_review = 0
        for r in pre.iter_pages(p, save=not args.no_save, output_dir=args.output_dir):
            n_review += r.applied["needs_review"]
        print(f"{p}: done in {time.perf_counter() - t0:.0f}s, {n_review} page(s) flagged for review")

