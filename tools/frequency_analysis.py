
"""
频率响应与谱图分析工具
用法示例：
  python tools/frequency_analysis.py --data_dir "./University of Ottawa Rolling-element Dataset – Vibration and Acoustic Faults under Constant Load and Speed conditions (UORED-VAFCLS)" --out_dir ./analysis_out
python tools/frequency_analysis.py --data_dir "./WTH" --out_dir ./analysis_out
python tools/frequency_analysis.py --data_dir "./BJTU_RAO_Bogie_Datasets" --out_dir ./analysis_out
功能：
- 读取数据集若干振动 CSV 信号，计算并保存谱图（PSD / spectrogram），标注主要频率带
- 读取 `BackboneCNN.CNN` 的第一卷积层权重并计算每个输出通道的幅频响应（C-FR），以及平均的总体幅频响应（O-FR），并绘图
- 读取 `DualModal_TFN` 的 TFconv_Gabor 第一层权重并计算对应的 C-FR / O-FR 并绘图

注意：要分析训练好的模型，请通过 `--backbone_ckpt` 与 `--dual_ckpt` 指定 checkpoint（可选）。
"""
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
import torch
import sys
import math

# ensure project root is on sys.path so `from Models...` works when running as script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# debug torch module
try:
    print('[debug] torch module file:', getattr(torch, '__file__', None))
    print('[debug] torch version:', getattr(torch, '__version__', None))
except Exception as e:
    print('[debug] torch introspect failed:', e)

def find_csv_files(data_dir, max_files=8):
    pattern = os.path.join(data_dir, '**', '*.csv')
    files = glob.glob(pattern, recursive=True)
    # debug prints to help locate why no files found
    print(f'[debug] glob pattern: {pattern}')
    print(f'[debug] matches found: {len(files)}')
    if len(files) == 0:
        # try common subfolder used in this repo
        alt = os.path.join(data_dir, '1_CSV_Raw_Data_Files (.csv)')
        print(f'[debug] trying alt dir: {alt} exists={os.path.isdir(alt)}')
        if os.path.isdir(alt):
            pattern2 = os.path.join(alt, '**', '*.csv')
            files = glob.glob(pattern2, recursive=True)
            print(f'[debug] alt pattern: {pattern2}, matches: {len(files)}')
    return files[:max_files]

def load_signals_from_csv(files):
    sigs = []
    print(f'[debug] attempting to load up to {len(files)} files (showing first 10):')
    for idx, f in enumerate(files):
        if idx < 10:
            print('  ', f)
        arr = None
        # try plain numpy load
        try:
            arr = np.loadtxt(f, delimiter=',')
            method = 'np.loadtxt'
        except Exception:
            # try skipping header
            try:
                arr = np.genfromtxt(f, delimiter=',', skip_header=1)
                method = 'np.genfromtxt(skip_header=1)'
            except Exception:
                # try pandas if available
                try:
                    import pandas as pd
                    df = pd.read_csv(f)
                    # try to pick first numeric column
                    num_col = None
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            num_col = col
                            break
                    if num_col is None:
                        # coerce first column to numeric
                        arrc = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                        arr = arrc.dropna().values
                    else:
                        arr = df[num_col].dropna().values
                    method = 'pandas'
                except Exception as e:
                    print(f'[debug] failed to load {f} with all methods: {e}')
                    arr = None
        if arr is None:
            print(f'[debug] file skipped: {f}')
            continue
        try:
            if isinstance(arr, np.ndarray) and arr.ndim > 1:
                arr = arr[:, 0]
            sigs.append(arr.astype(np.float32))
            print(f'[debug] loaded {f} via {method}, length={len(arr)}')
        except Exception as e:
            print(f'[debug] post-process failed for {f}: {e}')
            continue
    print(f'[debug] successfully loaded signals: {len(sigs)}')
    return sigs

def compute_welch_avg(sigs, fs, nperseg=2048):
    psds = []
    for s in sigs:
        f, Pxx = signal.welch(s, fs=fs, nperseg=min(nperseg, len(s)))
        psds.append(Pxx)
    if len(psds) == 0:
        return None, None
    Pxx_mean = np.mean(np.vstack(psds), axis=0)
    return f, Pxx_mean

def spectrogram_plot(sig, fs, outpath, nperseg=1024, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # smooth spectrogram in frequency dimension for nicer ridge shapes
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    Sxx_db_smooth = ndimage.gaussian_filter1d(Sxx_db, sigma=1.0, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(t, f, Sxx_db_smooth, shading='gouraud', cmap='magma')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('PSD (dB)')

    # add a small inset axes showing the mean PSD across time (for clearer peak curves)
    try:
        mean_psd = np.mean(Sxx, axis=1)
        mean_psd_db = 10 * np.log10(mean_psd + 1e-12)
        axins = fig.add_axes([0.72, 0.55, 0.22, 0.35])
        axins.plot(mean_psd_db, f, color='white', linewidth=1.2)
        axins.set_xlabel('PSD (dB)', fontsize=8)
        axins.set_ylabel('Freq [Hz]', fontsize=8)
        axins.tick_params(axis='both', which='major', labelsize=8)
        axins.invert_xaxis()
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def spectrogram_plot_with_bands(sig, fs, outpath, mark_bands=None, nperseg=1024, noverlap=None):
    # wrapper that highlights frequency bands on the spectrogram (horizontal spans)
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    Sxx_db_smooth = ndimage.gaussian_filter1d(Sxx_db, sigma=1.0, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(t, f, Sxx_db_smooth, shading='gouraud', cmap='magma')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('PSD (dB)')

    # highlight frequency bands as horizontal spans
    if mark_bands is not None:
        for (lo, hi) in mark_bands:
            try:
                ax.axhspan(lo, hi, color='orange', alpha=0.18)
                ax.axhline((lo + hi) / 2.0, color='orange', alpha=0.25, linestyle=':')
            except Exception:
                pass

    # inset mean PSD
    try:
        mean_psd = np.mean(Sxx, axis=1)
        mean_psd_db = 10 * np.log10(mean_psd + 1e-12)
        axins = fig.add_axes([0.72, 0.55, 0.22, 0.35])
        axins.plot(mean_psd_db, f, color='white', linewidth=1.2)
        axins.set_xlabel('PSD (dB)', fontsize=8)
        axins.set_ylabel('Freq [Hz]', fontsize=8)
        axins.tick_params(axis='both', which='major', labelsize=8)
        axins.invert_xaxis()
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def get_conv1d_kernels_conv_weight(weight_tensor):
    # weight_tensor: numpy array shape (out_channels, in_channels, kernel_size)
    return weight_tensor

def kernel_freq_response(kernels, fs, nfft=4096):
    # kernels: (out, in, k) -> compute per out-channel averaged across in
    out, inn, k = kernels.shape
    frs = []
    for i in range(out):
        # average across input channels to represent this output channel
        ker = np.mean(kernels[i], axis=0)
        pad = np.zeros(nfft)
        pad[:k] = ker[::-1]  # reverse kernel to match conv -> frequency response
        K = np.fft.rfft(pad)
        amp = np.abs(K)
        frs.append(amp)
    frs = np.vstack(frs)  # (out, nfft/2+1)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    ofr = np.mean(frs, axis=0)
    return freqs, frs, ofr

def plot_channel_and_overall(freqs, frs, ofr, title, outpath, mark_bands=None):
    plt.figure(figsize=(8,5))
    # convert to dB and smooth for nicer curve shapes
    frs_db = 20 * np.log10(frs + 1e-12)
    ofr_db = 20 * np.log10(ofr + 1e-12)

    # smooth overall and channel responses
    try:
        ofr_db_s = ndimage.gaussian_filter1d(ofr_db, sigma=2.0)
        frs_db_s = ndimage.gaussian_filter1d(frs_db, sigma=1.0, axis=1)
    except Exception:
        ofr_db_s = ofr_db
        frs_db_s = frs_db

    # plot C-FR (light lines) per channel
    for i in range(frs_db_s.shape[0]):
        plt.plot(freqs, frs_db_s[i], linestyle='-', color='#888888', alpha=0.35, linewidth=1)

    # plot O-FR (bold)
    plt.plot(freqs, ofr_db_s, color='#1f77b4', linewidth=2.5)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude (dB)')
    plt.title(title)
    if mark_bands is not None:
        for (lo, hi) in mark_bands:
            plt.axvspan(lo, hi, color='orange', alpha=0.15)
            plt.axvline((lo+hi)/2.0, color='orange', alpha=0.3, linestyle=':')
    plt.xlim(0, freqs.max())
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def get_prominent_bands(freqs, psd, topk=3, width_hz=50):
    # find peaks in PSD and return bands [peak-width/2, peak+width/2]
    peaks, _ = signal.find_peaks(psd)
    if len(peaks) == 0:
        return []
    peak_vals = psd[peaks]
    idx = np.argsort(peak_vals)[-topk:][::-1]
    bands = []
    for i in idx:
        p = peaks[i]
        center = freqs[p]
        bands.append((max(0, center-width_hz/2), center+width_hz/2))
    return bands

def load_backbone_model(device, ckpt=None, mid_channel=16, out_channels=10):
    from Models.BackboneCNN import CNN
    model = CNN(in_channels=1, out_channels=out_channels)
    if ckpt is not None and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd)
    model.to(device)
    return model

def load_dual_model(device, ckpt=None, mid_channel=16, out_channels=10, use_decision_fusion=False):
    from Models.TFN import DualModal_TFN
    model = DualModal_TFN(in_channels=1, out_channels=out_channels, mid_channel=mid_channel, use_decision_fusion=use_decision_fusion)
    if ckpt is not None and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd)
    model.to(device)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', default='./analysis_out')
    parser.add_argument('--fs', type=float, default=21000.0, help='sampling rate in Hz')
    parser.add_argument('--n_signals', type=int, default=6)
    parser.add_argument('--backbone_ckpt', type=str, default=None)
    parser.add_argument('--dual_ckpt', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # collect all CSV files once and group them by class keywords
    all_files = glob.glob(os.path.join(args.data_dir, '**', '*.csv'), recursive=True)
    if len(all_files) == 0:
        # try common alt folder
        alt = os.path.join(args.data_dir, '1_CSV_Raw_Data_Files (.csv)')
        all_files = glob.glob(os.path.join(alt, '**', '*.csv'), recursive=True)

    if len(all_files) == 0:
        print('No CSV signals found under', args.data_dir)
        return

    # class keyword mapping (robust matching)
    class_map = {
        'H': ['healthy', '1_healthy', '1_healthy'],
        'I': ['inner', 'inner_race', 'inner-race', 'innerrace'],
        'O': ['outer', 'outer_race', 'outer-race', 'outerrace'],
        'B': ['ball', 'rolling', 'ball_bearing', 'ball-bearing', 'rolling_element'],
        'C': ['cage', 'bearing_cage', 'cage']
    }

    files_by_class = {k: [] for k in class_map}
    for f in all_files:
        fl = f.lower()
        placed = False
        for k, kws in class_map.items():
            for kw in kws:
                if kw in fl:
                    files_by_class[k].append(f)
                    placed = True
                    break
            if placed:
                break
    # fallback: if some classes empty, try parent folder name matching
    for f in all_files:
        if any(f in v for v in files_by_class.values()):
            continue
        parent = os.path.basename(os.path.dirname(f)).lower()
        for k, kws in class_map.items():
            for kw in kws:
                if kw in parent:
                    files_by_class[k].append(f)
                    break

    # ensure we have at least one file per class; if missing, take any remaining files
    remaining = [f for f in all_files if not any(f in v for v in files_by_class.values())]
    for k in files_by_class:
        if len(files_by_class[k]) == 0 and remaining:
            files_by_class[k].append(remaining.pop(0))

    # compute per-class PSD and prominent bands, and plot spectrograms per class
    bands_by_class = {}
    for k, flist in files_by_class.items():
        sel = flist[: min(len(flist), args.n_signals)]
        sigs = load_signals_from_csv(sel)
        if len(sigs) == 0:
            print(f'[warn] no signals for class {k}, skipping')
            continue
        # compute average PSD for this class
        freqs_psd_c, psd_mean_c = compute_welch_avg(sigs, fs=args.fs)
        if freqs_psd_c is None:
            bands_c = []
        else:
            bands_c = get_prominent_bands(freqs_psd_c, psd_mean_c, topk=1, width_hz=200)
        bands_by_class[k] = bands_c

        # save PSD plot per class
        plt.figure(figsize=(8,4))
        if freqs_psd_c is not None:
            plt.semilogy(freqs_psd_c, psd_mean_c)
            for lo, hi in bands_c:
                plt.axvspan(lo, hi, color='orange', alpha=0.2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD')
        plt.title(f'Average PSD (class {k})')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f'psd_class_{k}.png'), dpi=200)
        plt.close()

        # spectrogram of first signal for this class with highlighted bands
        spectrogram_sig = sigs[0]
        spectrogram_plot_with_bands(spectrogram_sig, args.fs, os.path.join(args.out_dir, f'spectrogram_class_{k}.png'), mark_bands=bands_c)

    # aggregate all bands for FR plotting (flatten)
    all_bands = []
    for v in bands_by_class.values():
        all_bands.extend(v)

    # load models and plot FRs (use aggregated class bands for highlighting)
    device = 'cpu'
    backbone = load_backbone_model(device, ckpt=args.backbone_ckpt)
    w = backbone.layer1[0].weight.detach().cpu().numpy()  # (out, in, k)
    freqs, frs, ofr = kernel_freq_response(w, fs=args.fs)
    plot_channel_and_overall(freqs, frs, ofr, 'Backbone-CNN first conv FR', os.path.join(args.out_dir, 'backbone_firstconv_fr.png'), mark_bands=all_bands)

    dual = load_dual_model(device, ckpt=args.dual_ckpt)
    try:
        funconv = dual.vib_branch.funconv
        # case 1: legacy TFconv_Gabor who stores real/imag in weight
        if hasattr(funconv, 'weight') and funconv.weight is not None:
            w_t = funconv.weight.detach().cpu().numpy()
            real = w_t[0]
            imag = w_t[1]
            complex_kernels = real + 1j * imag
            mag_kernels = np.abs(complex_kernels)
        # case 2: TFconv_Gabor2D using superparams (f0, sigma, theta)
        elif hasattr(funconv, 'superparams') and funconv.superparams is not None:
            sp = funconv.superparams.detach().cpu().numpy()  # (out, in, 3)
            outc, inc, p = sp.shape
            # kernel size: try to read attribute, fallback to mid-channel-based guess
            k = int(getattr(funconv, 'kernel_size', getattr(funconv, 'funckernel_size',  (sp.shape[-1] if sp.ndim>2 else 15))))
            half = k // 2
            coords = np.linspace(-half, half, k)
            Y, X = np.meshgrid(coords, coords)
            mag_kernels = np.zeros((outc, inc, k))
            for i in range(outc):
                for j in range(inc):
                    f0 = float(sp[i, j, 0])
                    sigma = float(sp[i, j, 1])
                    theta = float(sp[i, j, 2]) if p > 2 else 0.0
                    c = math.cos(theta); s = math.sin(theta)
                    Xp = X * c + Y * s
                    Yp = -X * s + Y * c
                    denom = max(sigma * half, 1e-6)
                    gauss = np.exp(-(Xp**2 + Yp**2) / (2.0 * denom**2))
                    real_k = gauss * np.cos(2 * math.pi * f0 * Xp)
                    imag_k = gauss * np.sin(2 * math.pi * f0 * Xp)
                    complex_k = real_k + 1j * imag_k
                    mag2d = np.abs(complex_k)
                    # collapse frequency axis by mean to obtain 1D temporal kernel
                    mag_kernels[i, j, :] = np.mean(mag2d, axis=0)
        else:
            raise RuntimeError('funconv has neither weight nor superparams to compute FR')

        freqs2, frs2, ofr2 = kernel_freq_response(mag_kernels, fs=args.fs)
        plot_channel_and_overall(freqs2, frs2, ofr2, 'DualModal_TFN TFconv (Gabor) FR', os.path.join(args.out_dir, 'dual_tfconv_fr.png'), mark_bands=all_bands)

        # 如果 funconv 有 superparams（TFconv_Gabor2D），保存 superparams（包含 theta）供后续分析
        try:
            if hasattr(funconv, 'superparams') and funconv.superparams is not None:
                sp = funconv.superparams.detach().cpu().numpy()  # (out, in, p)
                np.savez(os.path.join(args.out_dir, 'dual_tfconv_superparams.npz'), superparams=sp)
                # 写出 theta（度）为可读文本，按 out_channel 行，in_channel 列
                if sp.shape[-1] > 2:
                    theta_deg = (sp[:, :, 2] * 180.0 / np.pi)
                    theta_path = os.path.join(args.out_dir, 'dual_tfconv_theta_degrees.txt')
                    with open(theta_path, 'w', encoding='utf-8') as fh:
                        fh.write('# theta (degrees) per out_channel x in_channel\n')
                        for i in range(theta_deg.shape[0]):
                            row = ','.join([f'{v:.6f}' for v in theta_deg[i]])
                            fh.write(row + '\n')
        except Exception:
            pass
    except Exception as e:
        print('failed to compute TFconv FR:', e)

    # write a short method description file explaining how spectrogram/PSD were computed
    try:
        with open(os.path.join(args.out_dir, 'spectrogram_method.txt'), 'w', encoding='utf-8') as fh:
            fh.write('Spectrogram / PSD computation method:\n')
            fh.write('\n')
            fh.write('Spectrogram: computed by short-time Fourier transform (STFT) using scipy.signal.spectrogram.\n')
            fh.write('For a signal x(t), the STFT is: STFT(t,f) = \n')
            fh.write('  S(t,f) = \sum_{n=0}^{N-1} x[n] w[n - t] e^{-j 2\pi f n / Fs} , where w is the window.\n')
            fh.write('Power spectrogram Sxx(t,f) = |S(t,f)|^2. We convert to dB via 10*log10(Sxx).\n')
            fh.write('\n')
            fh.write('Welch PSD: computed by scipy.signal.welch which averages periodograms of windowed, overlapping segments:\n')
            fh.write('  Pxx(f) = (1/K) \sum_{k=1}^K |FFT(x_k * w)|^2 / (Fs * U) , approximating the PSD.\n')
            fh.write('\n')
            fh.write('We detect prominent frequency bands by finding peaks in the averaged Pxx and creating a band of fixed width (default 200 Hz) around each peak.\n')
    except Exception:
        pass

    print('analysis saved to', os.path.abspath(args.out_dir))

if __name__ == '__main__':
    main()
