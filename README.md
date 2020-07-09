## Title
Automated Extraction ID Card


### Author
- Adila Alfa Krisnadhi
- Naufal Ihsan Pratama

### Stack
1. Tesseract-OCR
2. OpenCV
3. Python Data Science Library
4. Keras
5. Tensorflow

### Installation
```bash
brew install tesseract
rm /usr/local/share/tessdata/*.traineddata
cp traineddata/best/* /usr/local/share/tessdata
virtualenv venv
source venv/bin/activate
python app.py
```

### Research
[Github](https://www.github.com/naufalihsan/tesseract)
[API Docs](https://documenter.getpostman.com/view/6304914/SzS8rjNP)


### Resources
- https://tesseract-ocr.github.io/
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html