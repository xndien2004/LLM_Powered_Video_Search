# Application Structure

This application is divided into two main components:

- **Frontend**: Contains the following directories:
  - `statics`
  - `templates`
  
- **Backend**: Includes all other files in the project.

---

## Media Data Structure

Ensure all files are organized according to the structure below:

```
media/
├── contexts_bin/
│   ├── npz/
│   │   ├── sparse_context_matrix_asr.npz
│   │   ├── sparse_context_matrix_caption.npz
│   │   ├── sparse_context_matrix_number_tag.npz
│   │   ├── sparse_context_matrix_number.npz
│   │   ├── sparse_context_matrix_ocr.npz
│   │   ├── sparse_context_matrix_synthetic.npz
│   │   ├── sparse_context_matrix_tag.npz
│   ├── pkl/
│   │   ├── tfidf_transform_asr.pkl
│   │   ├── tfidf_transform_caption.pkl
│   │   ├── tfidf_transform_number_tag.pkl
│   │   ├── tfidf_transform_number.pkl
│   │   ├── tfidf_transform_ocr.pkl
│   │   ├── tfidf_transform_synthetic.pkl
│   │   ├── tfidf_transform_tag.pkl
├── faiss/
│   ├── faiss_openclip.bin
│   ├── ...
├── Keyframes/
│   ├── Keyframes_L01/keyframes/
│   │   ├── L01_V001/
│   │   │   ├── 001.jpg
│   │   │   ├── 002.jpg
│   │   │   ├── ...
│   ├── ...
├── map-keyframes/
│   ├── L01_V001.csv
│   ├── L01_V002.csv
│   ├── ...
├── media-info/
│   ├── L01_V001.json
│   ├── L01_V002.json
│   ├── ...
├── tag/
│   ├── tag_corpus.txt
├── id2img_fps_extra.json
├── id2img_fps.json
├── id2video.json
├── map-asr.json
```
 