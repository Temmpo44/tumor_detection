Bu depo araştırma/demonstrasyon amaçlıdır; tıbbi cihaz değildir. Nihai klinik kararlar yalnızca uzman hekimlerce verilir.

Genel Bakış

Görev: 4 sınıf MRI sınıflandırma — glioma, meningioma, pituitary, notumor

Modeller: PyTorch (ensemble uyumlu), opsiyonel YOLO ile kutu çizimi, opsiyonel canlı DICOM izleme

Ortam: Windows için optimize log ve bellek kullanımı

Sonuçlar (Örnek)

Doğruluk (Accuracy): 0.97

Macro F1: 0.97

Sınıf bazında:

glioma: P=0.98, R=0.95, F1=0.96

meningioma: P=0.93, R=0.94, F1=0.93

notumor: P=0.98, R=1.00, F1=0.99

pituitary: P=0.98, R=0.97, F1=0.97

Veri Kümesi

Kaynak: Kaggle – Brain Tumor MRI Dataset (4-class) (glioma, meningioma, pituitary, notumor)

Not: Veri Kaggle kullanım şartlarına uygun olarak indirildi ve yalnızca araştırma amaçlı kullanıldı.

Bağlantı yeri: (Kendi Kaggle linkini buraya ekle)

Gereksinimler
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn numpy pandas matplotlib seaborn
# İsteğe bağlı:
pip install ultralytics opencv-python pydicom watchdog

Dizin Yapısı (öneri)
project/
├─ data/train/{glioma,meningioma,pituitary,notumor}/
├─ data/val/{glioma,meningioma,pituitary,notumor}/
├─ pytorch_models_windows/
├─ scripts/ (train.py, evaluate.py, infer.py, yolo_draw_pred_vs_true.py, live_mri_yolo_viewer.py, run_video_infer.py)
├─ assets/ (confusion_matrix.png, pred_vs_true_examples.png)
└─ README.md

Eğitim
python scripts/train.py \
  --train_dir data/train \
  --val_dir data/val \
  --out_dir pytorch_models_windows \
  --epochs 20 --batch_size 32 --img_size 224 \
  --optimizer adamw --lr 3e-4 --seed 42

Değerlendirme
python scripts/evaluate.py \
  --val_dir data/val \
  --weights pytorch_models_windows/best.pth \
  --img_size 224 \
  --report_out assets/cls_report.txt \
  --cm_out assets/confusion_matrix.png

İnferans

Tek görsel

python scripts/infer.py \
  --image path/to/sample.png \
  --weights pytorch_models_windows/best.pth \
  --img_size 224


Video → kare → vaka tahmini (opsiyonel)

python scripts/run_video_infer.py \
  --frames_dir path/to/frames \
  --ckpt pytorch_models_windows/best.pth \
  --img_size 224 \
  --classes "glioma,meningioma,notumor,pituitary" \
  --out video_infer_out.json

(Opsiyonel) YOLO ile Kutu Çizimi
yolo task=detect mode=train model=yolov8n.pt data=brain_tumor_yolo.yaml imgsz=640 epochs=60 batch=16

python scripts/yolo_draw_pred_vs_true.py \
  --weights runs/detect/train/weights/best.pt \
  --img_dir data/val/images \
  --label_dir data/val/labels \
  --out_dir assets/yolo_vis

(Opsiyonel) Canlı DICOM İzleme (PoC)
python scripts/live_mri_yolo_viewer.py \
  --watch DICOM_INBOX \
  --weights runs/detect/train/weights/best.pt \
  --imgsz 640 --conf 0.25 --show 1
# GUI yoksa: --show 0

İpuçları

Val transform’ları deterministik olsun; shuffle=False, drop_last=False.

Değerlendirmede model.eval() + torch.no_grad() kullan.

Sınıf-özel eşik/kalibrasyon (temperature scaling) ile glioma↔meningioma karışıklığını azalt.

TTA (flip/rotate) ile küçük ama istikrarlı artış sağlanır.

Hasta-bazlı split ile veri sızıntısını önle.

Etik & Gizlilik

Görsellerde PHI barındırma.

Model çıktıları tanısal değildir; karar destek amaçlıdır.

Lisans

Kod: MIT (ya da seçtiğin lisans)

Veri: Kaggle veri setinin lisans/şartlarına tabidir.

Teşekkür

Kaggle Brain Tumor MRI Dataset

PyTorch, scikit-learn, Ultralytics YOLO, pydicom, OpenCV
