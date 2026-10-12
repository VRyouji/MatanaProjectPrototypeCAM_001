using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

/// <summary>
/// Mengubah gambar wajah menjadi embedding (vektor ciri) memakai model SFace (ONNX).
/// Kelas ini TIDAK tahu sumber gambarnya (webcam, upload, kamera aplikasi),
/// sehingga nanti bisa dipakai ulang saat kamera sudah terintegrasi di aplikasi.
/// </summary>
public sealed class FaceEmbedder : IDisposable
{
    private const int InputSize = 112;
    private readonly InferenceSession _session;
    private readonly string _inputName;

    public FaceEmbedder(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public float[] GetEmbedding(Mat image, Rect faceRect)
    {
        // Perluas area crop 15% supaya seluruh wajah masuk, lalu batasi ke ukuran gambar
        int mx = (int)(faceRect.Width * 0.15);
        int my = (int)(faceRect.Height * 0.15);
        int x1 = Math.Max(0, faceRect.X - mx);
        int y1 = Math.Max(0, faceRect.Y - my);
        int x2 = Math.Min(image.Width, faceRect.X + faceRect.Width + mx);
        int y2 = Math.Min(image.Height, faceRect.Y + faceRect.Height + my);
        var roi = new Rect(x1, y1, x2 - x1, y2 - y1);

        using var face = new Mat(image, roi);
        using var resized = new Mat();
        Cv2.Resize(face, resized, new Size(InputSize, InputSize));
        using var rgb = new Mat();
        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

        var bytes = new byte[InputSize * InputSize * 3];
        Marshal.Copy(rgb.Data, bytes, 0, bytes.Length);

        // Format input model: [1, 3, 112, 112], nilai piksel 0-255 tanpa normalisasi
        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });
        for (int y = 0; y < InputSize; y++)
            for (int x = 0; x < InputSize; x++)
                for (int c = 0; c < 3; c++)
                    tensor[0, c, y, x] = bytes[(y * InputSize + x) * 3 + c];

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
        using var results = _session.Run(inputs);
        return results.First().AsEnumerable<float>().ToArray();
    }

    public static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-9);
    }

    public void Dispose() => _session.Dispose();
}
