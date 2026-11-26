# Fine-tuning fusion DualTrack dengan JPG original MDPI

Pipeline ini membaca JPG grayscale **1030 × 750** dari folder original dan
melatih **fusion module + head translasi Y/Z**. Kedua encoder lengkap, termasuk
proyeksi fitur, positional embedding, BatchNorm dan attention, tetap dibekukan
dan berada dalam mode evaluasi. Tidak perlu mengulang pretraining encoder.

## Environment dan data yang sudah disiapkan

Jalankan dari root repository DualTrack dengan environment `dualtrack`:

```bash
conda activate dualtrack
cd /home/syahid/github/DualTrack
```

Dataset yang telah disiapkan berada di
`data/mdpi_original_dualtrack_v1/manifest.json`; laporan pemeriksaannya adalah
`data/mdpi_original_dualtrack_v1/report.json`. Folder `data` pada workspace ini
menunjuk ke `/mnt/Data-A/syahid/data`.

Manifest menunjuk JPG di
`/home/syahid/github/Sensorless_Small_DS/data/MDPI Compiled Data`.
Keberadaan sumber original tetap diperlukan saat training. Tidak ada salinan
gambar tambahan atau input gambar 160 × 120 dalam dataset baru ini.

| Split | Scan | Window 10 frame |
|---|---:|---:|
| Train | 59 | 262 |
| Validation | 13 | 72 |
| Test | 13 | 65 |

Cakupan tetap 5mm/10mm × V1/V2/V3. Assignment scan, kelompok dependensi,
eligibility dan window mengikuti metadata v4 yang sudah dibekukan. Dua scan
yang sebelumnya dilewati tetap tidak disertakan. Builder memakai HDF5 v4 hanya
untuk metadata; dataset `frames` yang beresolusi kecil tidak dibaca.

JPG `i.jpg` berpasangan dengan kolom `i` pada `coords_interpolated.npy`.
Translasi referensi adalah `(0, -Y_reference, X_reference)` dengan rotasi tetap.
Nomor frame yang tampil dalam CSV log mulai dari 1, sedangkan nama JPG mulai 0.
Builder memeriksa seluruh hash JPG asli/pixel decoded, NPY, log dan volume yang
menentukan kelompok dependensi. Volume NRRD tidak menjadi label atau input.

Untuk membangun dataset baru, pilih direktori output yang belum ada:

```bash
python scripts/data/prepare_mdpi_original.py --output data/mdpi_original_dualtrack_run2
```

`--source-root` dan `--reference-h5` tersedia untuk lokasi lain. Crop dapat
ditetapkan dengan `--local-crop X Y W H` dan `--global-roi X Y W H` sebelum
membangun manifest. Pilih preprocessing dari gambar train, kemudian bekukan.
Manifest lama dan sumber original tidak ditimpa.

## Input model

- Local encoder: native crop 256 × 256 pada `[x=387, y=247, w=256, h=256]`.
  Crop pusat ini dipilih setelah memeriksa contoh train dari keenam kombinasi.
- Global encoder: seluruh area gambar `[0, 0, 1030, 750]`, di-resize dengan
  `INTER_AREA` menjadi 224 × 224.
- Kedua input dinormalisasi `/255`. Tidak ada augmentasi geometris pada pilot.
- Dari 10 frame, global encoder melihat indeks `[0,2,4,6,8]`.

Crop lokal mempertahankan resolusi sumber. ViT spatial lokal pada checkpoint
memerlukan feature map yang sesuai dengan input 256 × 256; memasukkan seluruh
1030 × 750 memerlukan perubahan model. Transform piksel setelah crop/resize
disimpan pada `nominal_geometry` di manifest. Skala gambar bersifat nominal.

## Pemeriksaan dan training

```bash
# Hanya validasi sumber train/validation dan indeks dataset.
python scripts/finetune_mdpi_fusion.py --check-only

# Tambahkan satu forward pada window train.
python scripts/finetune_mdpi_fusion.py --check-only --check-weights

# Satu langkah optimisasi untuk memeriksa batas parameter yang berubah.
# Tidak menyimpan bobot hasil langkah tersebut.
python scripts/finetune_mdpi_fusion.py --smoke-step

# Mulai training penuh dengan konfigurasi default.
python scripts/finetune_mdpi_fusion.py --train
```

Default tanpa opsi mode adalah pemeriksaan data. `--report /path/new.json`
menyimpan bukti pemeriksaan/smoke step. Gunakan `--device cpu` bila diperlukan.

Konfigurasi ada di
`configs/dualtrack_finetunes/mdpi_original_fusion.yaml`. Override umum:

```bash
python scripts/finetune_mdpi_fusion.py --train \
  --manifest data/mdpi_original_dualtrack_v1/manifest.json \
  --output experiments/mdpi_original_fusion_run2 \
  --batch-size 4 --num-workers 2
```

Checkpoint sumber adalah `dualtrack_ft_tus_rec_2025_v3_best.pt`. Model 6D
dipulihkan dengan strict loading dan global positional embedding `absolute`.
Head kemudian menjadi `Linear(512,2)` yang diinisialisasi dari baris translasi
Y/Z head pretrained. Fusion tetap dimulai dari bobot pretrained.

Hanya **44.144.898 parameter** yang dilatih: fusion 44.143.872 dan head 1.026.
Optimizer AdamW memiliki dua grup: learning rate fusion `1e-5`, head `1e-4`,
weight decay `1e-4`. Default batch 4, maksimum 50 epoch, patience 8, min_delta
`1e-4`, gradient clipping 1.0 dan AMP pada CUDA. Learning rate mengikuti cosine
schedule dengan satu epoch awal pada learning rate awal.

Head menghasilkan sembilan perpindahan Y/Z. Penjumlahan membentuk sepuluh
posisi, kemudian selisih posisi membentuk 45 pasangan dalam urutan
`[(i,j) for j in range(1,10) for i in range(j)]`. Loss adalah MSE yang dibagi
**dua koordinat terawasi**; nilainya berbeda dari loss lama yang dibagi tiga.
Galat Euclidean pasangan tetap dapat dibandingkan pada referensi yang sama.

Window memakai frame eligible dengan gap indeks original maksimal 3. Tidak
ada interpolasi/smoothing label baru, pengisian frame invalid, atau padding
window. Seluruh anggota kelas gambar decoded duplikat dan gambar konstan
dikecualikan mengikuti aturan yang dibekukan.

## Output dan pemilihan checkpoint

Default output `experiments/mdpi_original_fusion_v1/` harus belum ada:

```text
config.json
provenance.json
epoch0_validation.json
history.jsonl
best_validation.json
checkpoints/best.pt
checkpoints/last.pt
trajectory_validation_best/
```

Pemilihan checkpoint memakai `mean_pair_error_nominal_mm` pada validation.
Epoch 0 juga menjadi kandidat terbaik. Seluruh perbaikan ketat menyimpan best;
patience direset jika perbaikan melampaui `min_delta`. Uji trajectory pada best
checkpoint tidak mengubah kriteria pemilihan. Gunakan `--skip-trajectory-eval`
jika hanya membutuhkan training.

Bobot/buffer kedua encoder diperiksa setiap epoch agar tetap identik. Bundle
checkpoint menyimpan seluruh bobot, optimizer, scheduler, scaler, RNG, generator
DataLoader, baseline train-only, konfigurasi dan hash manifest/pretrained/kode.
Loader menolak schema atau hash yang berbeda. Sumber original harus tetap
sesuai manifest. Perubahan preprocessing memerlukan dataset dan run baru.
GradScaler menangani overflow awal FP16 dengan melewati langkah tersebut dan
menurunkan skala; jumlahnya dicatat sebagai `amp_skipped_optimizer_steps`.
Run dihentikan bila seluruh langkah pada suatu epoch dilewati.

Run yang terputus dapat dilanjutkan ke **direktori output baru**:

```bash
python scripts/finetune_mdpi_fusion.py --train \
  --resume experiments/mdpi_original_fusion_v1/checkpoints/last.pt \
  --output experiments/mdpi_original_fusion_resumed
```

`best.pt` harus tersedia di samping checkpoint resume. Pengaturan training
harus tetap sama; run yang sudah mencapai batas epoch/patience tidak di-resume.

## Evaluasi

```bash
python scripts/evaluate_mdpi_fusion.py \
  --checkpoint experiments/mdpi_original_fusion_v1/checkpoints/best.pt \
  --output reports/mdpi_original_fusion_validation \
  --device cuda --image-controls
```

Default split adalah validation. Output berisi metrik window, `report.json`,
CSV per scan/segmen, `trajectories.npz`, dan grafik contoh trajectory.
Kontrol repeat-first, gambar kosong dan pengacakan waktu diterapkan seragam
pada kedua cabang untuk mengukur sensitivitas gambar; hasil kontrol ini bukan
akurasi pada akuisisi valid.

Evaluasi trajectory memakai window 10 frame dengan stride 9 dan window penuh
terakhir untuk menutup tail. Prediksi edge yang sama dirata-rata. Gap >3
memisahkan segmen; segmen <10 frame dilaporkan sebagai dilewati. Setiap segmen
hanya memiliki satu anchor awal tanpa fitting rotasi/skala atau pengisian gap
dari referensi. Report membedakan agregat seluruh segmen dan scan dengan satu
segmen eligible lengkap. Trajectory dievaluasi dalam float32.

Baseline diam dan rata-rata sembilan langkah dipasang dari train saja. Test
tidak di-infer otomatis; evaluasi test memerlukan `--split test --allow-test`
setelah semua pengaturan dikunci. Split ini berbagi tanggal akuisisi dan bukan
evaluasi sesi/phantom independen. Semua galat adalah **mm nominal terhadap
estimasi interpolated**, bukan pengukuran pose probe 6DoF terkalibrasi.

## Verifikasi implementasi

```bash
python -m unittest discover -s tests -p 'test_mdpi_*.py' -v
```

Tes mencakup pemetaan JPG/label dan hash, pembacaan metadata tanpa pixel HDF5,
split/window, arah dan integrasi pasangan, batas gradient/parameter/buffer,
strict checkpoint round-trip, stitching tail/gap, baseline train-only, serta
penolakan inference test tanpa opt-in.
