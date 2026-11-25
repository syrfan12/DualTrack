# Jalankan di Python Interactor 3D Slicer:
# exec(open("/home/syahid/github/DualTrack/scripts/inference/reconstruct_3dslicer.py").read())
import glob
import os

import h5py
import numpy as np
import vtk

# ---------- Config ----------
pred_h5 = "/home/syahid/github/DualTrack/data/screen_captures_conv/screen-captures_pred.h5"
images_dir = "/home/syahid/github/DualTrack/data/screen_captures"
out_nrrd = "/home/syahid/github/DualTrack/data/screen_captures_conv/screen-captures_recon.nrrd"
out_spacing = 0.2  # mm, isotropic. Kecilkan (mis. 0.1) bila memori cukup.

# ---------- Load prediction ----------
with h5py.File(pred_h5, "r") as f:
    glob_T = f["pred_tracking_glob"][:].astype(np.float64)  # (N-1,4,4), mm, frame i -> frame 0
    pixel_to_image = f["pixel_to_image"][:].astype(np.float64)  # pixel -> mm (sudah termasuk spacing)

files = sorted(glob.glob(os.path.join(images_dir, "*.png")))[: len(glob_T) + 1]
n = len(files)
assert n == len(glob_T) + 1, f"{n} gambar vs {len(glob_T)} transform"
transforms = [np.eye(4)] + list(glob_T)  # frame 0 = identitas

# ---------- Build sequence ----------
seq = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", "recon_seq")
seq.SetIndexName("frame")
seq.SetIndexUnit("")
tmp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")

for i, path in enumerate(files):
    reader = vtk.vtkPNGReader()
    reader.SetFileName(path)
    lum = vtk.vtkImageLuminance()
    lum.SetInputConnection(reader.GetOutputPort())
    flip = vtk.vtkImageFlip()  # baris 0 = atas, sesuai konvensi pixel_to_image
    flip.SetInputConnection(lum.GetOutputPort())
    flip.SetFilteredAxis(1)
    flip.Update()
    img = vtk.vtkImageData()
    img.DeepCopy(flip.GetOutput())
    img.SetOrigin(0, 0, 0)
    img.SetSpacing(1, 1, 1)
    tmp.SetAndObserveImageData(img)

    ijk_to_ras = vtk.vtkMatrix4x4()
    M = transforms[i] @ pixel_to_image  # spacing 0.037 mm sudah ada di pixel_to_image
    for r in range(4):
        for c in range(4):
            ijk_to_ras.SetElement(r, c, float(M[r, c]))
    tmp.SetIJKToRASMatrix(ijk_to_ras)
    seq.SetDataNodeAtValue(tmp, str(i))

slicer.mrmlScene.RemoveNode(tmp)

browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", "recon_browser")
browser.AddSynchronizedSequenceNode(seq)
slicer.modules.sequences.logic().UpdateAllProxyNodes()
proxy = browser.GetProxyNode(seq)

# ---------- Reconstruct ----------
recon = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLVolumeReconstructionNode")
recon.SetAndObserveInputSequenceBrowserNode(browser)
recon.SetAndObserveInputVolumeNode(proxy)
recon.SetOutputSpacing(out_spacing, out_spacing, out_spacing)
recon.SetFillHoles(True)

slicer.app.pauseRender()
slicer.modules.volumereconstruction.logic().ReconstructVolumeFromSequence(recon)
volume = recon.GetOutputVolumeNode()
volume.SetName("screen_captures_recon")
slicer.app.resumeRender()

roi = recon.GetInputROINode()
slicer.mrmlScene.RemoveNode(recon)
slicer.mrmlScene.RemoveNode(browser)
slicer.mrmlScene.RemoveNode(seq)
if roi:
    slicer.mrmlScene.RemoveNode(roi)

slicer.util.setSliceViewerLayers(background=volume, fit=True)
slicer.util.saveNode(volume, out_nrrd)
print("Selesai:", out_nrrd, "spacing:", volume.GetSpacing())
