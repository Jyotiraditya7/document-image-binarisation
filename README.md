# Document Image Binarisation

## Getting Started

1. **Fork this repository** to your own GitHub account.
2. **Clone your forked repository**:
    ```bash
    git clone https://github.com/your-username/document-image-binarisation.git
    cd document-image-binarisation
    ```

3. **Create and activate a Python virtual environment**:

    **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    **On Windows:**
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

4. **Install dependencies** (if any requirements file is provided):
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the main script**:
    ```bash
    cd ocr_seg
    python ocr_segm.py
    ```

## Notes

- Images are located in the `/images` directory.
- To use a different image, edit `ocr_segm.py` at line 3:
  ```python
  myImage = cv2.imread('images/shadow_side.jpg')
  ```
  Change the filename as needed.