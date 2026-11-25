# FIR Extraction Plan: Qwen3-VL (local, Ollama)

**Status:** plan only, not built yet. **Scope for now:** classify every page, extract data from **FIR pages only**.

The model is `qwen3-vl:32b`, served by Ollama on our own server. All data stays on our network.

---

## 1. Flow

```
PDF ──► image_processing.py  (cleaned RGB page images, in page order)
        │
        ▼  for each page, in order (sequential: the previous page's type must be known)
   ┌───────────────────────────────────────────────────────────────┐
   │ 1. Build prompt: rules + previous page's type + FIR keys       │
   │ 2. Send the final page image to Ollama qwen3-vl:32b            │
   │    (JSON-schema-enforced reply)                                │
   │ 3. Validate reply; retry once if invalid                       │
   │ 4. page_type = header found ? that type : previous page's type │  ← enforced in code
   │ 5. If FIR page: keep extracted values with their page number   │
   └───────────────────────────────────────────────────────────────┘
        │
        ▼
   Merge all FIR pages into one FIR record ──► {pdf}_extraction.json
```

- **Fixed pipeline, not a free agent.** Every step always runs in the same order, so each one can be inspected and rerun on its own. That makes it auditable and repeatable, which a self-directing agent is not.
- **One pass first:** each page gets one request that both classifies it and extracts its values.
- **Two passes only if needed:** if one-pass extraction proves weak, switch to classify all pages first, then extract the FIR pages (see step 5 of the build order).

---

## 2. One request per page

Input: the cleaned final page image (PNG-encoded in memory, lossless) plus the prompt. Output: JSON in a fixed structure, enforced with Ollama's `format` option:

```json
{
  "header_found": true,
  "header_text": "FIRST INFORMATION REPORT / प्रथम सूचना रिपोर्ट (I.I.F.-I)",
  "page_type": "FIR",
  "fields": { "district": "पाली", "police_station": "रानी", "fir_no": "2", "fir_date": "1-1-09", "...": null }
}
```

### Classification rules
A page starts a new section **only** if it has a printed form header:

| Header (any language / form code) | `page_type` |
|---|---|
| First Information Report / प्रथम सूचना रिपोर्ट / I.I.F.-I | `FIR` |
| Crime Details Form / अपराध विवरण फार्म / I.I.F.-II | `CRIME_DETAILS` |
| Arrest / Court Surrender / गिरफ्तारी / I.I.F.-III | `ARREST` |
| Property Search & Seizure / जब्ती / I.I.F.-IV | `SEIZURE` |
| Final Form / Final Report / अंतिम रिपोर्ट / I.I.F.-V | `FINAL_REPORT` |
| Medical / X-ray / injury / post-mortem report | `MEDICAL` |
| Any other printed header | `OTHER` |

- **A page without a header continues the previous page's type.** The code decides this from `header_found`, so a model slip on one page can't break the sequence. The first page with no header defaults to `OTHER` and is flagged for review.
- **Fields are filled only on FIR pages.** On every other page `fields` is left empty, which also keeps those requests short.

### Extraction rules (in the prompt)
- Copy every value **exactly as written. Never translate or transliterate.** Hindi stays in Devanagari, and dates and numbers are kept as written.
- Blank or not present → `null`. Present but unreadable → `"[illegible]"`. **Never guess.**
- Handwritten values take precedence over the printed label text next to them.

### Proposed FIR field keys (to confirm)
| Group | Keys |
|---|---|
| Header | `district`, `police_station`, `year`, `fir_no`, `fir_date` |
| Law | `act_sections` |
| Occurrence | `occurrence_day`, `occurrence_date_from`, `occurrence_date_to`, `occurrence_time_from`, `occurrence_time_to` |
| Information received | `info_received_date`, `info_received_time`, `gd_entry_no`, `gd_entry_time`, `type_of_information` |
| Place | `place_direction_distance`, `beat_no`, `place_address` |
| Complainant | `complainant_name`, `complainant_father_husband_name`, `complainant_dob_age`, `complainant_nationality`, `complainant_occupation`, `complainant_address` |
| Case | `accused_details`, `delay_reason`, `properties_stolen`, `total_value`, `inquest_report` |
| Narrative & action | `fir_contents`, `action_taken`, `officer_name`, `officer_rank` |

### Merging FIR pages
- Every value records its **source page**.
- If the same key is found on two pages with different values, both are kept in `conflicts` for review. Nothing is overwritten silently.
- `fir_contents` (the narrative) usually runs over several pages, so its parts are joined in page order.

---

## 3. Settings (one place, configurable)

| Setting | Start value | Note |
|---|---|---|
| `host` | `http://<server>:11434` | Ollama reachable only on our network |
| `model` | `qwen3-vl:32b` | |
| `temperature` / `seed` | `0` / fixed | repeatable answers |
| `num_ctx` | 16k–32k | measure: one full page image + answer |
| image | final page, full resolution, PNG | scale down only if too slow or over context; compare accuracy |
| `timeout` / retries | per-page timeout, 1 retry | invalid JSON → retry, then flag |
| needs-review pages | later: also send the original image | compare accuracy |

- **Data security:** no internet calls. Images travel in the request body and are not stored on the server.

---

## 4. Outputs

| File | Content |
|---|---|
| `{pdf}_extraction.json` | page list (`page`, `page_type`, `header_text`, `inherited`) + merged FIR record (`value`, `page`, `conflicts`) |
| `{pdf}_{page}_ai.json` | raw model reply per page (audit / debugging) |

**Code:** a new file `extraction.py`, used like this:

```python
from extraction import FIRExtractor

result = FIRExtractor(host="http://<server>:11434").extract("doc.pdf", save=True, output_dir="out")
```

Inside, it calls `image_processing.DocumentPreprocessor().process(pdf, return_all=True)` and sends the images from memory.

`app.ipynb` will show each page with its type and the extracted FIR values.

---

## 5. Build order

1. **Connection check:** send one page to Ollama and confirm a valid JSON reply. Measure time per page and the context size needed.
2. **Classification:** run it on the 4 sample PDFs (71 pages) and compare page types with the correct labels (drafted from the samples, corrected by the team).
3. **Extraction:** run it on the FIR pages and compare with correct values for 1–2 cases.
4. **Measure:** classification accuracy, per-field exact match (plus a loose match that ignores spacing and punctuation), and time per page.
5. **Decide:**
   - keep one pass;
   - or move to two passes: classify all pages first, then extract only the FIR pages, possibly sending all FIR pages of a case in one request;
   - adjust the prompt, image size, and whether the original image is sent as well.

---

## 6. Open items (needed before building)

1. **Server:** Ollama address and port; can the Mac reach it directly or only through an SSH tunnel?
2. **Ollama version** on the server (`ollama --version`). Qwen3-VL needs a recent version.
3. **Field keys:** confirm, add or remove keys in section 2, or rename them.
4. **Correct values** for the FIR pages of 1–2 sample cases, to measure accuracy.
