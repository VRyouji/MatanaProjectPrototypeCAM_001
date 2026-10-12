using OpenCvSharp;

// ============================================================
// Prototype Face Recognition - Sistem Absensi Sales
// Alur: foto referensi -> embedding -> kamera live -> cocokkan
// ============================================================

const string CascadePath         = "models/haarcascade_frontalface_alt.xml";
const string RecognizerModelPath = "models/face_recognition_sface_2021dec.onnx";
const string ReferenceDir        = "reference_faces";
const int    CameraIndex         = 0;      // ganti ke 1 jika kamera default bukan yang dimaksud
const double MatchThreshold      = 0.363;  // titik awal; sesuaikan setelah melihat skor nyata di layar

// 1. Pastikan file model tersedia
foreach (var path in new[] { CascadePath, RecognizerModelPath })
{
    if (!File.Exists(path))
    {
        Console.WriteLine($"[ERROR] File tidak ditemukan: {path}");
        Console.WriteLine("Lihat README.md untuk cara download.");
        return;
    }
}

using var cascade  = new CascadeClassifier(CascadePath);
using var embedder = new FaceEmbedder(RecognizerModelPath);

// 2. Muat semua foto referensi (nama file = nama orang)
var references = new List<(string Name, float[] Embedding)>();

if (!Directory.Exists(ReferenceDir))
    Directory.CreateDirectory(ReferenceDir);

var photoFiles = Directory.GetFiles(ReferenceDir)
    .Where(f => new[] { ".jpg", ".jpeg", ".png" }.Contains(Path.GetExtension(f).ToLower()));

foreach (var file in photoFiles)
{
    using var img = Cv2.ImRead(file);
    if (img.Empty()) { Console.WriteLine($"[SKIP] Tidak bisa membaca {file}"); continue; }

    var faces = DetectFaces(img);
    if (faces.Length == 0) { Console.WriteLine($"[SKIP] Tidak ada wajah di {file}"); continue; }

    // Ambil wajah terbesar di foto
    var main = faces.OrderByDescending(r => r.Width * r.Height).First();
    var name = Path.GetFileNameWithoutExtension(file);
    references.Add((name, embedder.GetEmbedding(img, main)));
    Console.WriteLine($"[OK] Referensi dimuat: {name}");
}

if (references.Count == 0)
{
    Console.WriteLine($"[ERROR] Belum ada foto referensi valid di folder '{ReferenceDir}'.");
    Console.WriteLine("Letakkan foto wajah (mis. vanesson.jpg) lalu jalankan ulang.");

    return;
}

// 3. Buka kamera lokal
using var capture = new VideoCapture(CameraIndex);
if (!capture.IsOpened())
{
    Console.WriteLine("[ERROR] Kamera tidak bisa dibuka. Coba ubah CameraIndex atau tutup aplikasi lain yang memakai kamera.");
    return;
}

Console.WriteLine("Kamera aktif. Tekan 'Q' di jendela kamera untuk keluar.");

using var frame = new Mat();
while (true)
{
    capture.Read(frame);
    if (frame.Empty()) break;

    foreach (var rect in DetectFaces(frame))
    {
        var live = embedder.GetEmbedding(frame, rect);

        string bestName = "";
        double bestScore = -1;
        foreach (var (name, refEmbedding) in references)
        {
            double score = FaceEmbedder.CosineSimilarity(live, refEmbedding);
            if (score > bestScore) { bestScore = score; bestName = name; }
        }

        bool isMatch = bestScore >= MatchThreshold;
        var color = isMatch ? Scalar.LimeGreen : Scalar.Red;
        var label = isMatch ? $"{bestName} ({bestScore:F2})" : $"Tidak dikenal ({bestScore:F2})";

        Cv2.Rectangle(frame, rect, color, 2);
        Cv2.PutText(frame, label, new Point(rect.X, Math.Max(rect.Y - 8, 15)),
            HersheyFonts.HersheySimplex, 0.6, color, 2);
    }

    Cv2.ImShow("Prototype Face Recognition - tekan Q untuk keluar", frame);
    if (Cv2.WaitKey(1) == 'q') break;
}

Cv2.DestroyAllWindows();

// ------------------------- Helper -------------------------

Rect[] DetectFaces(Mat image)
{
    using var gray = new Mat();
    Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);
    Cv2.EqualizeHist(gray, gray);
    return cascade.DetectMultiScale(gray, 1.1, 5, minSize: new Size(80, 80));
}
