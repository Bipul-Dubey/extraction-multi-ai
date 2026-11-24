# FIR Documents Extractor & OpenCV Preprocessing Pipeline

An interactive AI-powered application built as a Jupyter Notebook (`app.ipynb`) with an OpenCV preprocessing pipeline designed to clean, enhance, and extract structured legal data from police First Information Report (FIR) documents (PDFs, images, or raw text).

---

## 🔍 Features

- **High-DPI PDF Page Rendering**:
  - **300 DPI**: Default high-resolution rendering for standard printed documents.
  - **400 DPI**: Optimized resolution for faint ink, dense stamps, and **small handwriting**.
- **OpenCV Document Preprocessing**:
  - **Illumination & Shadow Correction**: Morphological background division to remove scanner shadows and paper yellowing.
  - **CLAHE Contrast Enhancement**: Enhances faint ballpoint pen strokes and carbon copy text.
  - **Bilateral Filtering**: Denoises while strictly preserving delicate handwritten strokes.
  - **Automatic Deskewing**: Calculates text orientation via bounding moments and straightens tilted documents.
  - **Adaptive Gaussian Binarization**: Produces crisp black & white text representations for OCR.
- **Multimodal LLM Extraction**: Sends the cleaned images directly to Gemini for structured JSON schema extraction.
- **Interactive Gradio UI**: Embedded directly inside the notebook with side-by-side OpenCV before/after image previews.

---

## 📁 Project Structure

- [app.ipynb](file:///Users/atomic/Desktop/experiment/fir-docs-extractor/app.ipynb): Main interactive notebook with OpenCV preprocessing and Gradio interface.
- [requirements.txt](file:///Users/atomic/Desktop/experiment/fir-docs-extractor/requirements.txt): Python dependencies specifying the latest versions.
- [.env.example](file:///Users/atomic/Desktop/experiment/fir-docs-extractor/.env.example): Template for configuring your `GEMINI_API_KEY`.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Notebook

```bash
jupyter notebook app.ipynb
```

