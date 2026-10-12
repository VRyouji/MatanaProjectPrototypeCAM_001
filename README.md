# Prototype Face Recognition (Absensi Sales)

## Kebutuhan
- Windows 10/11, .NET 8 SDK
- Webcam

## File yang dibutuhkan di folder `models/`
1. `face_recognition_sface_2021dec.onnx`
   Dari OpenCV Zoo: github.com/opencv/opencv_zoo -> models/face_recognition_sface
2. `haarcascade_frontalface_default.xml`
   Dari OpenCV: github.com/opencv/opencv -> data/haarcascades
   (buka file-nya, klik "Raw" / ikon download, simpan sebagai .xml)

(Model YuNet tidak lagi dipakai.)

## Setup
1. Taruh foto wajah referensi di `reference_faces/`, contoh `vanesson.jpg`.
   Foto harus jelas dan menghadap depan. Nama file = nama yang tampil.
2. Pastikan package terpasang: `dotnet restore`
3. Jalankan dari folder project: `dotnet run`
4. Tekan `Q` di jendela kamera untuk keluar.

## Cara baca hasil
- Kotak hijau + nama = cocok dengan foto referensi
- Kotak merah = tidak dikenal
- Angka dalam kurung = skor kemiripan (cosine). Cocok jika >= MatchThreshold (di Program.cs)
