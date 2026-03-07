import { useState, useEffect, useRef, useCallback } from "react";

const API = "https://compact-cities-narrow-cooperative.trycloudflare.com";

// File size threshold above which we warn the user (50MB)
const LARGE_FILE_BYTES = 50 * 1024 * 1024;

const C = {
  bg: "#060A0F",
  surface: "#0D1520",
  border: "#162030",
  accent: "#00E5A0",
  warn: "#FFB800",
  danger: "#FF4560",
  text: "#C8D8E8",
  muted: "#4A6070",
  white: "#EAF4FF",
};

const FAVICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="6" fill="#0D1520"/>
  <path d="M10 8 C6 8 4 11 4 14 C4 18 6 21 10 22 C12 22.5 13 21 14 19 C15 17 15 15 16 14 C17 13 18 13 19 14 C20 15 20 18 21 20 C22 22 24 23 26 21 C28 19 28 15 26 12 C24 9 21 8 18 8 C15 8 13 9 12 10 C11 9 10.5 8 10 8Z" fill="#00E5A0" opacity="0.9"/>
  <line x1="16" y1="12" x2="16" y2="20" stroke="#060A0F" stroke-width="1.2" opacity="0.7"/>
  <line x1="12" y1="16" x2="20" y2="16" stroke="#060A0F" stroke-width="1.2" opacity="0.7"/>
</svg>`;

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #060A0F;
    color: #C8D8E8;
    font-family: 'Outfit', sans-serif;
    font-size: 14px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #060A0F; }
  ::-webkit-scrollbar-thumb { background: #162030; border-radius: 3px; }

  @keyframes pulse-border {
    0%, 100% { border-color: #00E5A044; }
    50%       { border-color: #00E5A0BB; }
  }
  @keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes spin  { to { transform: rotate(360deg); } }
  @keyframes flash { 0%,100% { opacity:1 } 50% { opacity:0.3 } }

  .fade-up { animation: fadeUp 0.45s ease both; }

  .scanline-overlay {
    pointer-events: none;
    position: fixed; inset: 0;
    overflow: hidden;
    z-index: 9999;
  }
  .scanline-overlay::after {
    content: '';
    display: block;
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(transparent, #00E5A018, transparent);
    animation: scanline 8s linear infinite;
  }
`;

// ── Detect Firefox for folder-upload warning ─────────────────────────────────
const IS_FIREFOX = typeof navigator !== "undefined" &&
  navigator.userAgent.toLowerCase().includes("firefox");

// ── Utility components ────────────────────────────────────────────────────────

function Mono({ children, color, size = 12 }) {
  return (
    <span style={{
      fontFamily: "'DM Mono', monospace", fontSize: size,
      color: color || C.muted, letterSpacing: "0.02em",
    }}>
      {children}
    </span>
  );
}

function Tag({ label, color = C.accent }) {
  return (
    <span style={{
      fontFamily: "'DM Mono', monospace", fontSize: 10,
      color, border: `1px solid ${color}55`,
      padding: "2px 8px", borderRadius: 2,
      letterSpacing: "0.1em", textTransform: "uppercase",
    }}>
      {label}
    </span>
  );
}

function ProgressBar({ value = 0, color = C.accent }) {
  return (
    <div style={{ background: `${color}18`, borderRadius: 2, height: 6, overflow: "hidden" }}>
      <div style={{
        height: "100%", width: `${value}%`,
        background: `linear-gradient(90deg, ${color}88, ${color})`,
        borderRadius: 2, transition: "width 0.35s ease",
        boxShadow: value > 0 ? `0 0 8px ${color}55` : "none",
      }} />
    </div>
  );
}

function Spinner({ color = C.accent, size = 16 }) {
  return (
    <div style={{
      width: size, height: size,
      border: `2px solid ${color}22`,
      borderTop: `2px solid ${color}`,
      borderRadius: "50%",
      animation: "spin 0.8s linear infinite",
      flexShrink: 0,
    }} />
  );
}

function DurationBadge({ seconds }) {
  if (seconds == null) return null;
  return (
    <span style={{
      fontFamily: "'DM Mono', monospace", fontSize: 10,
      color: C.muted, border: `1px solid ${C.border}`,
      padding: "2px 7px", borderRadius: 2, letterSpacing: "0.06em",
    }}>
      {seconds}s
    </span>
  );
}

function Alert({ message, color = C.danger }) {
  if (!message) return null;
  return (
    <div style={{
      background: `${color}0F`, border: `1px solid ${color}44`,
      borderRadius: 4, padding: "10px 14px",
      display: "flex", gap: 8, alignItems: "flex-start",
    }}>
      <span style={{ color, flexShrink: 0 }}>⚠</span>
      <span style={{ color, fontSize: 12, lineHeight: 1.6 }}>{message}</span>
    </div>
  );
}

// ── Metric grid ───────────────────────────────────────────────────────────────

function MetricBlock({ result, stageKey }) {
  const items =
    stageKey === "stage0" ? [
      ["Slices extracted", result.slices_extracted],
      ["CT window (HU)", result.window_hu],
      ["Input format", result.format],
      ["Status", "ready"],
    ] : stageKey === "stage1" ? [
      ["Detections found", result.detections_found],
      ["Detection rate", `${(result.detection_rate * 100).toFixed(1)}%`],
      ["mAP@0.5", result.map_at_50],
      ["Precision", result.precision],
    ] : stageKey === "stage2" ? [
      ["Crops segmented", result.crops_segmented],
      ["Mean Dice", result.mean_dice],
      ["Max Dice", result.max_dice],
      ["Mean IoU", result.mean_iou],
    ] : stageKey === "stage3" ? [
      ["Prediction", result.prediction],
      ["Confidence", `${(result.confidence * 100).toFixed(1)}%`],
      ["Model accuracy", result.model_accuracy],
      ["AUC", result.auc],
    ] : stageKey === "stage4" ? [
      ["Top feature", result.top_features?.[0]],
      ["YOLO attribution", "computed"],
      ["U-Net attribution", "computed"],
      ["EfficientNet attr", "computed"],
    ] : [];

  return (
    <div style={{
      marginTop: 14, paddingTop: 12,
      borderTop: `1px solid ${C.border}`,
      display: "grid", gridTemplateColumns: "1fr 1fr",
      gap: "6px 16px",
    }}>
      {items.map(([k, v]) => (
        <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
          <Mono>{k}</Mono>
          <Mono color={C.accent} size={12}>{String(v)}</Mono>
        </div>
      ))}
    </div>
  );
}

// ── Stage card ────────────────────────────────────────────────────────────────

function StageCard({ stage, data, index }) {
  const status = data?.status || "idle";
  const progress = data?.progress || 0;
  const duration = data?.duration_s;

  const statusColor = {
    idle: C.muted, running: C.accent, complete: C.accent,
    error: C.danger, queued: C.muted,
  }[status] || C.muted;

  return (
    <div className="fade-up" style={{
      animationDelay: `${index * 60}ms`,
      background: C.surface,
      border: `1px solid ${status === "running" ? C.accent + "55" : C.border}`,
      borderRadius: 6, padding: "16px 20px",
      animation: status === "running"
        ? "pulse-border 1.8s ease infinite"
        : `fadeUp 0.45s ${index * 60}ms ease both`,
      transition: "border-color 0.3s",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <span style={{
          fontSize: 18,
          filter: status === "running" ? `drop-shadow(0 0 6px ${C.accent})` : "none",
        }}>
          {stage.icon}
        </span>
        <span style={{ color: C.white, fontWeight: 600, fontSize: 13, flex: 1 }}>
          {stage.label}
        </span>
        {status === "running" && <Spinner />}
        {status === "complete" && <DurationBadge seconds={duration} />}
        {status === "complete" && <Tag label="complete" color={C.accent} />}
        {status === "error" && <Tag label="error" color={C.danger} />}
      </div>

      <p style={{ color: C.muted, fontSize: 12, marginBottom: 12, lineHeight: 1.6 }}>
        {stage.desc}
      </p>

      {/* Show progress bar for running/complete, idle bar for queued */}
      <ProgressBar value={progress} color={statusColor} />

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
        <Mono>
          {status === "idle" ? "waiting…" :
            status === "queued" ? "queued…" :
              status === "error" ? data?.message || "error" :
                data?.message || ""}
        </Mono>
        <Mono color={statusColor}>
          {status === "idle" ? "" : `${progress}%`}
        </Mono>
      </div>

      {/* Error message surfaced clearly */}
      {status === "error" && data?.message && (
        <div style={{
          marginTop: 10,
          background: `${C.danger}0F`, border: `1px solid ${C.danger}33`,
          borderRadius: 4, padding: "8px 12px",
        }}>
          <Mono color={C.danger} size={11}>{data.message}</Mono>
        </div>
      )}

      {status === "complete" && data?.result && (
        <MetricBlock result={data.result} stageKey={stage.key} />
      )}
    </div>
  );
}

// ── Test case picker ──────────────────────────────────────────────────────────

function CasePicker({ onSelect, disabled }) {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [fetchErr, setFetchErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/cases`)
      .then(r => r.json())
      .then(data => { setCases(data); setLoading(false); })
      .catch(() => {
        setFetchErr("Could not load test cases. Is the backend running on port 5000?");
        setLoading(false);
      });
  }, []);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "16px 0" }}>
      <Spinner size={14} /><Mono>Loading test cases…</Mono>
    </div>
  );

  if (fetchErr) return <Alert message={fetchErr} />;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <p style={{ color: C.muted, fontSize: 12, marginBottom: 6, lineHeight: 1.6 }}>
        Select one of the five pre-loaded KiTS21 test cases held out from training.
        Cases are from the raw dataset — the full pipeline including preprocessing runs on each.
      </p>

      {cases.map(c => {
        const isSelected = selected === c.case_id;
        const isM = c.label === "Malignant";
        const unavailable = c.available === false;

        return (
          <div
            key={c.case_id}
            onClick={() => !unavailable && setSelected(c.case_id)}
            style={{
              background: isSelected ? `${C.accent}0D` : C.surface,
              border: `1px solid ${isSelected ? C.accent + "88" : C.border}`,
              borderRadius: 6, padding: "14px 16px",
              cursor: unavailable ? "not-allowed" : "pointer",
              opacity: unavailable ? 0.45 : 1,
              transition: "all 0.2s",
              display: "flex", alignItems: "center", gap: 14,
            }}
          >
            <div style={{
              width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
              background: isM ? C.danger : C.accent,
              boxShadow: `0 0 6px ${isM ? C.danger : C.accent}88`,
            }} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                <Mono size={12} color={C.white}>{c.case_id}</Mono>
                <Tag label={c.label} color={isM ? C.danger : C.accent} />
                {unavailable && <Tag label="not downloaded" color={C.muted} />}
              </div>
              <p style={{ color: C.muted, fontSize: 11, lineHeight: 1.5 }}>
                {c.description}
              </p>
            </div>
            <div style={{ textAlign: "right", flexShrink: 0 }}>
              <Mono size={10}>
                {typeof c.slice_count === "number" ? `${c.slice_count} slices` : c.slice_count}
              </Mono>
              <div style={{ marginTop: 2 }}>
                <Mono size={10}>Dice {c.dice_3d}</Mono>
              </div>
            </div>
          </div>
        );
      })}

      <button
        onClick={() => selected && onSelect(selected)}
        disabled={!selected || disabled}
        style={{
          marginTop: 8, padding: "13px",
          background: selected && !disabled ? C.accent : C.border,
          color: selected && !disabled ? C.bg : C.muted,
          border: "none", borderRadius: 4,
          fontFamily: "'DM Mono', monospace",
          fontSize: 13, fontWeight: 500, letterSpacing: "0.08em",
          cursor: selected && !disabled ? "pointer" : "not-allowed",
          textTransform: "uppercase", transition: "all 0.2s",
        }}
      >
        RUN SELECTED CASE
      </button>
    </div>
  );
}

// ── Upload zone ───────────────────────────────────────────────────────────────

function UploadZone({ onStart, disabled, uploadProgress }) {
  const [files, setFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  const [sizeWarn, setSizeWarn] = useState(null);
  const niftiRef = useRef(null);
  const dicomRef = useRef(null);
  const pngRef = useRef(null);

  const handleFiles = useCallback((incoming) => {
    const arr = Array.from(incoming);

    // NIfTI — direct file or inside a case folder
    const nii = arr.find(f =>
      f.name.match(/\.nii(\.gz)?$/i) ||
      f.webkitRelativePath?.match(/imaging\.nii(\.gz)?$/i)
    );

    const dcm = arr.filter(f => f.name.match(/\.dcm$/i));
    const pngs = arr.filter(f => f.name.match(/\.png$/i));

    if (nii) {
      setFiles([nii]);
      setError(null);
      setSizeWarn(
        nii.size > LARGE_FILE_BYTES
          ? `Large file detected (~${Math.round(nii.size / 1024 / 1024)}MB) — upload may take 30–60 seconds on slower connections.`
          : null
      );
    } else if (dcm.length > 0) {
      setFiles(dcm);
      setError(null);
      setSizeWarn(null);
    } else if (pngs.length >= 5) {
      setFiles(pngs);
      setError(null);
      setSizeWarn(
        pngs.length > 200
          ? `${pngs.length} PNG slices selected — upload may take a moment.`
          : null
      );
    } else {
      setError(
        "No recognised files found. Expected a NIfTI file (.nii / .nii.gz), " +
        "a case folder with imaging.nii.gz, DICOM files (.dcm), or " +
        "a folder of pre-processed PNG slices (at least 5)."
      );
    }
  }, []);

  const fmt =
    files.length === 1 && files[0].name.match(/\.nii/) ? "NIfTI" :
      files.length > 0 && files[0].name.match(/\.dcm$/i) ? "DICOM" :
        files.length >= 5 && files[0].name.match(/\.png$/i) ? "PNG Slices" : null;

  return (
    <div>
      {/* Format guidance */}
      <div style={{
        background: `${C.accent}08`, border: `1px solid ${C.border}`,
        borderRadius: 6, padding: "12px 16px", marginBottom: 14,
        display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 16px",
      }}>
        {[
          ["NIfTI file", "Browse to imaging.nii.gz inside any case folder — works for KiTS21 raw cases"],
          ["DICOM folder", ".dcm files — standard clinical scanner output, converted automatically"],
          ["PNG slices", "Upload the images/ folder from processed/slices/test/case_XXXXX/"],
          ["Processing", "CT windowing [-79, 304] HU applied automatically for NIfTI and DICOM"],
        ].map(([label, desc]) => (
          <div key={label}>
            <Mono size={10} color={C.accent}>{label}</Mono>
            <p style={{ color: C.muted, fontSize: 11, marginTop: 2, lineHeight: 1.5 }}>
              {desc}
            </p>
          </div>
        ))}
      </div>

      {/* Drop zone — drag and drop only. No onClick to avoid double-popup bug. */}
      <div
        onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        style={{
          border: `2px dashed ${dragOver ? C.accent : C.border}`,
          borderRadius: 8, padding: "28px 24px",
          textAlign: "center",
          transition: "all 0.25s",
          background: dragOver ? `${C.accent}08` : "transparent",
        }}
      >
        <div style={{ fontSize: 26, marginBottom: 8, opacity: 0.5 }}>⬡</div>
        <p style={{ color: files.length ? C.accent : C.text, fontWeight: 600, marginBottom: 6 }}>
          {files.length
            ? (files.length === 1 ? files[0].name : `${files.length} files selected`)
            : "Drag and drop files here"}
        </p>
        <Mono size={11}>NIfTI · DICOM · PNG slices</Mono>

        {/* Explicit browse button for single NIfTI file */}
        <div style={{ marginTop: 14 }}>
          <button
            onClick={(e) => { e.stopPropagation(); niftiRef.current?.click(); }}
            style={{
              padding: "7px 18px", background: "transparent",
              border: `1px solid ${C.border}`, color: C.muted,
              borderRadius: 3, fontFamily: "'DM Mono', monospace",
              fontSize: 11, cursor: "pointer",
              letterSpacing: "0.05em", textTransform: "uppercase",
              transition: "all 0.2s",
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = C.accent; e.currentTarget.style.color = C.accent; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = C.border; e.currentTarget.style.color = C.muted; }}
          >
            Browse NIfTI file (.nii / .nii.gz / imaging.nii.gz)
          </button>
        </div>

        {/* NIfTI single file — accept .gz so imaging.nii.gz is visible in OS picker */}
        <input ref={niftiRef} type="file" accept=".nii,.gz"
          style={{ display: "none" }}
          onChange={e => handleFiles(e.target.files)} />

        {/* DICOM folder input */}
        <input ref={dicomRef} type="file" multiple
          webkitdirectory=""
          style={{ display: "none" }}
          onChange={e => handleFiles(e.target.files)} />

        {/* PNG slice folder input */}
        <input ref={pngRef} type="file" multiple
          webkitdirectory=""
          style={{ display: "none" }}
          onChange={e => handleFiles(e.target.files)} />
      </div>

      {fmt && (
        <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 8 }}>
          <Tag label={fmt} color={C.accent} />
          {fmt === "DICOM" && (
            <Mono size={11} color={C.muted}>
              {files.length} files · will be converted to NIfTI via dcm2niix
            </Mono>
          )}
          {fmt === "PNG Slices" && (
            <Mono size={11} color={C.muted}>
              {files.length} slices · preprocessing stage will be skipped
            </Mono>
          )}
        </div>
      )}

      {/* Folder buttons — stopPropagation prevents drop zone from firing */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 10 }}>
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (IS_FIREFOX) {
              setError(
                "Firefox does not support folder uploads. " +
                "Please use Chrome, Edge, or Brave — or upload individual files."
              );
              return;
            }
            dicomRef.current?.click();
          }}
          style={folderBtnStyle}
          onMouseEnter={e => Object.assign(e.currentTarget.style, folderBtnHover)}
          onMouseLeave={e => Object.assign(e.currentTarget.style, folderBtnStyle)}
        >
          📁  DICOM Folder
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (IS_FIREFOX) {
              setError(
                "Firefox does not support folder uploads. " +
                "Please use Chrome, Edge, or Brave."
              );
              return;
            }
            pngRef.current?.click();
          }}
          style={folderBtnStyle}
          onMouseEnter={e => Object.assign(e.currentTarget.style, folderBtnHover)}
          onMouseLeave={e => Object.assign(e.currentTarget.style, folderBtnStyle)}
        >
          🖼  PNG Slices Folder
        </button>
      </div>

      {/* Warnings and errors */}
      {sizeWarn && (
        <div style={{ marginTop: 10 }}>
          <Alert message={sizeWarn} color={C.warn} />
        </div>
      )}
      {error && (
        <div style={{ marginTop: 10 }}>
          <Alert message={error} color={C.danger} />
        </div>
      )}

      {/* Upload progress bar — shown while uploading or server is processing */}
      {uploadProgress !== null && (
        <div style={{ marginTop: 12 }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            alignItems: "baseline", marginBottom: 4,
          }}>
            <Mono size={10} color={C.muted}>
              {uploadProgress < 100 ? "UPLOADING" : "PROCESSING…"}
            </Mono>
            {uploadProgress < 100 && (
              <Mono size={11} color={C.accent}>{uploadProgress}%</Mono>
            )}
          </div>
          <div style={{
            height: 5, background: `${C.border}66`,
            borderRadius: 3, overflow: "hidden",
          }}>
            <div style={{
              height: "100%", borderRadius: 3,
              width: uploadProgress < 100 ? `${uploadProgress}%` : "100%",
              background: uploadProgress < 100 ? C.accent : C.warn,
              transition: "width 0.2s ease",
              animation: uploadProgress === 100 ? "pulse 1.2s ease infinite" : "none",
            }} />
          </div>
        </div>
      )}

      {/* Submit */}
      <button
        onClick={() => files.length && !uploadProgress && onStart(files)}
        disabled={!files.length || disabled || uploadProgress !== null}
        style={{
          marginTop: 12, width: "100%", padding: "14px",
          background: files.length && !disabled && uploadProgress === null ? C.accent : C.border,
          color: files.length && !disabled && uploadProgress === null ? C.bg : C.muted,
          border: "none", borderRadius: 4,
          fontFamily: "'DM Mono', monospace",
          fontSize: 13, fontWeight: 500, letterSpacing: "0.08em",
          cursor: files.length && !disabled && uploadProgress === null ? "pointer" : "not-allowed",
          textTransform: "uppercase", transition: "all 0.2s",
        }}
      >
        {uploadProgress !== null
          ? uploadProgress < 100 ? `UPLOADING… ${uploadProgress}%` : "PROCESSING…"
          : "INITIATE PIPELINE ANALYSIS"}
      </button>
    </div>
  );
}

const folderBtnStyle = {
  padding: "10px",
  background: "transparent",
  border: `1px solid #162030`,
  color: "#4A6070",
  borderRadius: 4,
  fontFamily: "'DM Mono', monospace",
  fontSize: 11, cursor: "pointer",
  letterSpacing: "0.05em", textTransform: "uppercase",
  transition: "all 0.2s", width: "100%",
};

const folderBtnHover = {
  ...folderBtnStyle,
  borderColor: "#00E5A0",
  color: "#00E5A0",
};

// ── Summary panel ─────────────────────────────────────────────────────────────

function SummaryPanel({ summary }) {
  if (!summary) return null;
  const m = summary.key_metrics;
  const t = summary.stage_timings || {};
  const isM = m.classification === "Malignant";

  const timeVals = [t.stage0_s, t.stage1_s, t.stage2_s, t.stage3_s].filter(Boolean);
  const totalTime = timeVals.reduce((a, b) => a + b, 0).toFixed(2);

  return (
    <div className="fade-up" style={{
      background: C.surface,
      border: `1px solid ${isM ? C.danger + "66" : C.accent + "66"}`,
      borderRadius: 6, padding: 24, marginTop: 24,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: isM ? C.danger : C.accent,
          boxShadow: `0 0 8px ${isM ? C.danger : C.accent}`,
          animation: "flash 2s ease infinite",
        }} />
        <span style={{ color: C.white, fontWeight: 700, fontSize: 15 }}>
          ANALYSIS COMPLETE — WRITTEN SUMMARY
        </span>
      </div>

      {/* Big metrics */}
      <div style={{
        background: `${isM ? C.danger : C.accent}0F`,
        border: `1px solid ${isM ? C.danger : C.accent}33`,
        borderRadius: 4, padding: "14px 18px", marginBottom: 18,
        display: "flex", flexWrap: "wrap", gap: 20,
      }}>
        {[
          ["CLASSIFICATION", m.classification.toUpperCase(), isM ? C.danger : C.accent, 24],
          ["CONFIDENCE", m.confidence_pct, C.white, 24],
          ["CERTAINTY", m.certainty_level.toUpperCase(), C.warn, 20],
          ["DICE SCORE", String(m.dice_score), C.text, 24],
        ].map(([label, value, color, size], i) => (
          <div key={label} style={{
            borderLeft: i > 0 ? `1px solid ${C.border}` : "none",
            paddingLeft: i > 0 ? 20 : 0,
          }}>
            <Mono size={10} color={C.muted}>{label}</Mono>
            <div style={{
              color, fontWeight: 700, fontSize: size,
              fontFamily: "'DM Mono', monospace", marginTop: 2,
            }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      <p style={{ color: C.text, lineHeight: 1.8, fontSize: 13, marginBottom: 16 }}>
        {summary.case_summary}
      </p>

      {/* Timing breakdown */}
      {timeVals.length > 0 && (
        <div style={{
          background: `${C.border}44`, border: `1px solid ${C.border}`,
          borderRadius: 4, padding: "10px 14px", marginBottom: 14,
          display: "flex", gap: 20, flexWrap: "wrap", alignItems: "center",
        }}>
          <Mono size={10} color={C.muted}>STAGE TIMINGS</Mono>
          {[
            ["PRE-PROC", t.stage0_s],
            ["YOLO", t.stage1_s],
            ["U-Net", t.stage2_s],
            ["EffNet", t.stage3_s],
          ].filter(([, v]) => v != null).map(([label, val]) => (
            <div key={label} style={{ display: "flex", gap: 5, alignItems: "baseline" }}>
              <Mono size={10}>{label}</Mono>
              <Mono size={11} color={C.accent}>{val}s</Mono>
            </div>
          ))}
          <div style={{
            borderLeft: `1px solid ${C.border}`, paddingLeft: 14,
            display: "flex", gap: 5, alignItems: "center",
          }}>
            <Mono size={10}>TOTAL</Mono>
            <Mono size={11} color={C.warn}>{totalTime}s</Mono>
          </div>
        </div>
      )}

      <div style={{
        background: `${C.warn}0A`, border: `1px solid ${C.warn}33`,
        borderRadius: 4, padding: "10px 14px",
      }}>
        <p style={{ color: C.warn, fontSize: 11, lineHeight: 1.7 }}>
          ⚠ {summary.disclaimer}
        </p>
      </div>
    </div>
  );
}

function Lightbox({ src, label, caption, onClose }) {
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [drag, setDrag] = useState(null);

  // close on Escape
  useEffect(() => {
    const handler = e => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const handleWheel = e => {
    e.preventDefault();
    setScale(s => Math.min(8, Math.max(1, s - e.deltaY * 0.001)));
  };

  const handleMouseDown = e => {
    if (scale <= 1) return;
    setDrag({ startX: e.clientX - pos.x, startY: e.clientY - pos.y });
  };

  const handleMouseMove = e => {
    if (!drag) return;
    setPos({ x: e.clientX - drag.startX, y: e.clientY - drag.startY });
  };

  const handleMouseUp = () => setDrag(null);

  const resetZoom = () => { setScale(1); setPos({ x: 0, y: 0 }); };

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, zIndex: 9999,
        background: "rgba(0,0,0,0.88)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
    >
      {/* close button */}
      <button
        onClick={onClose}
        style={{
          position: "fixed", top: 18, right: 22,
          background: "none", border: "none", color: "#fff",
          fontSize: 26, cursor: "pointer", lineHeight: 1, zIndex: 10000,
        }}
      >✕</button>

      {/* zoom controls */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          position: "fixed", bottom: 22, left: "50%", transform: "translateX(-50%)",
          display: "flex", gap: 10, alignItems: "center", zIndex: 10000,
        }}
      >
        <button onClick={() => setScale(s => Math.min(8, s + 0.5))}
          style={lbBtnStyle}>＋</button>
        <span style={{ color: "#aaa", fontSize: 12, minWidth: 42, textAlign: "center" }}>
          {Math.round(scale * 100)}%
        </span>
        <button onClick={() => setScale(s => Math.max(1, s - 0.5))}
          style={lbBtnStyle}>－</button>
        <button onClick={resetZoom} style={lbBtnStyle}>Reset</button>
      </div>

      {/* image */}
      <div
        onClick={e => e.stopPropagation()}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          cursor: scale > 1 ? (drag ? "grabbing" : "grab") : "default",
          userSelect: "none",
        }}
      >
        <img
          src={src}
          alt={label}
          style={{
            maxWidth: "88vw", maxHeight: "82vh",
            objectFit: "contain", display: "block",
            transform: `scale(${scale}) translate(${pos.x / scale}px, ${pos.y / scale}px)`,
            transformOrigin: "center center",
            transition: drag ? "none" : "transform 0.1s ease",
          }}
          draggable={false}
        />
      </div>

      {/* caption bar */}
      <div style={{
        position: "fixed", bottom: 62, left: "50%", transform: "translateX(-50%)",
        background: "rgba(6,10,15,0.85)", borderRadius: 4, padding: "5px 14px",
        pointerEvents: "none",
      }}>
        <Mono size={11} color={C.accent}>{label}</Mono>
        <span style={{ color: "#666", fontSize: 10, marginLeft: 8 }}>{caption}</span>
      </div>
    </div>
  );
}

const lbBtnStyle = {
  background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.15)",
  color: "#fff", borderRadius: 4, padding: "3px 10px", cursor: "pointer", fontSize: 13,
};

// ── SHAP Analysis component ───────────────────────────────────────────────────

function ShapAnalysis({ jobId, result, shapStatus }) {
  const [lightbox, setLightbox] = useState(null);
  if (shapStatus !== "complete" || !result) return null;

  const zones = result.zones || [];
  const top3 = result.top3 || [];
  const written = result.written_summary || "";
  const src = `${API}/image/${jobId}/stage4?t=${Date.now()}`;

  // Map zone names to 3x3 grid positions (row-major)
  const ZONE_ORDER = [
    "Upper-Left", "Upper-Centre", "Upper-Right",
    "Mid-Left", "Central", "Mid-Right",
    "Lower-Left", "Lower-Centre", "Lower-Right",
  ];
  const zoneByName = Object.fromEntries(zones.map(z => [z.name, z]));
  const maxScore = Math.max(...zones.map(z => z.score), 0.001);

  // Colour interpolation: low = dark blue, high = warm red/orange
  const zoneColor = score => {
    const t = score / maxScore;
    if (t < 0.33) return `rgba(0,100,200,${0.2 + t * 0.6})`;
    if (t < 0.66) return `rgba(255,140,0,${0.3 + t * 0.5})`;
    return `rgba(255,60,60,${0.5 + t * 0.5})`;
  };

  return (
    <>
      {lightbox && (
        <Lightbox src={lightbox.src} label={lightbox.label}
          caption={lightbox.caption} onClose={() => setLightbox(null)} />
      )}

      <div className="fade-up" style={{
        background: C.surface, border: `1px solid ${C.accent}55`,
        borderRadius: 6, padding: "18px 20px", marginTop: 16,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
          <span style={{ fontSize: 18 }}>◇</span>
          <span style={{ color: C.white, fontWeight: 600, fontSize: 13 }}>
            SHAP Attribution Analysis
          </span>
          <Mono size={10} color={C.muted}>— gradient saliency · EfficientNet-B0</Mono>
        </div>

        {/* Top row: saliency image + 3x3 zone grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 18 }}>

          {/* Saliency image */}
          <div style={{
            background: C.bg, border: `1px solid ${C.border}`,
            borderRadius: 5, overflow: "hidden", cursor: "zoom-in",
          }}
            onClick={() => setLightbox({
              src,
              label: "Stage 4 — SHAP Saliency",
              caption: "Gradient saliency — brighter = more influential",
            })}
          >
            <div style={{
              width: "100%", aspectRatio: "1 / 1", background: "#08111A",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <img src={src} alt="SHAP saliency"
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
                onError={e => { e.currentTarget.style.display = "none"; }}
              />
            </div>
            <div style={{ padding: "7px 10px", borderTop: `1px solid ${C.border}` }}>
              <Mono size={10} color={C.accent}>Stage 4 — SHAP Saliency</Mono>
              <p style={{ color: C.muted, fontSize: 10, marginTop: 2 }}>
                Blue = low influence · Red = high influence · Click to expand
              </p>
            </div>
          </div>

          {/* 3x3 zone grid */}
          <div>
            <Mono size={10} color={C.muted} style={{ marginBottom: 8, display: "block" }}>
              ZONE ATTRIBUTION MAP
            </Mono>
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
              gap: 3, marginBottom: 12,
            }}>
              {ZONE_ORDER.map(name => {
                const z = zoneByName[name] || { score: 0, pct: 0 };
                const isTop = top3.some(t => t.name === name);
                return (
                  <div key={name} style={{
                    background: zoneColor(z.score),
                    border: `1px solid ${isTop ? C.warn : C.border}`,
                    borderRadius: 3, padding: "6px 4px",
                    textAlign: "center",
                  }}>
                    <div style={{
                      color: "#fff", fontSize: 9,
                      fontFamily: "'DM Mono', monospace", lineHeight: 1.3,
                    }}>
                      {name.replace("-", "\n")}
                    </div>
                    <div style={{
                      color: isTop ? C.warn : C.text,
                      fontSize: 11, fontFamily: "'DM Mono', monospace",
                      fontWeight: isTop ? 700 : 400, marginTop: 3,
                    }}>
                      {z.pct}%
                    </div>
                  </div>
                );
              })}
            </div>
            <Mono size={9} color={C.muted}>
              ◈ highlighted border = top-3 zone
            </Mono>
          </div>
        </div>

        {/* Top-3 bar chart */}
        <div style={{
          background: C.bg, border: `1px solid ${C.border}`,
          borderRadius: 4, padding: "12px 14px", marginBottom: 16,
        }}>
          <Mono size={10} color={C.muted} style={{ display: "block", marginBottom: 10 }}>
            TOP-3 CONTRIBUTING ZONES
          </Mono>
          {top3.map((z, i) => (
            <div key={z.name} style={{ marginBottom: i < 2 ? 10 : 0 }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "baseline", marginBottom: 4,
              }}>
                <Mono size={10}>{z.name}</Mono>
                <Mono size={11} color={i === 0 ? C.warn : C.accent}>{z.pct}%</Mono>
              </div>
              <div style={{
                height: 6, background: `${C.border}66`, borderRadius: 3, overflow: "hidden",
              }}>
                <div style={{
                  height: "100%", borderRadius: 3,
                  width: `${z.pct}%`,
                  background: i === 0 ? C.warn : i === 1 ? C.accent : C.text,
                  transition: "width 0.6s ease",
                }} />
              </div>
            </div>
          ))}
        </div>

        {/* Written radiologist summary */}
        <div style={{
          background: `${C.accent}08`, border: `1px solid ${C.accent}22`,
          borderRadius: 4, padding: "12px 14px",
        }}>
          <Mono size={10} color={C.accent} style={{ display: "block", marginBottom: 8 }}>
            RADIOLOGIST INTERPRETATION
          </Mono>
          <p style={{ color: C.text, fontSize: 12, lineHeight: 1.8, margin: 0 }}>
            {written}
          </p>
        </div>
      </div>
    </>
  );
}

// ── Image gallery ─────────────────────────────────────────────────────────────

function ImageGallery({ jobId, isComplete }) {
  if (!isComplete) return null;
  const [lightbox, setLightbox] = useState(null);

  const stages = [
    { key: "stage1", label: "Stage 1 — Detection", caption: "YOLO bounding boxes on representative CT slice" },
    { key: "stage2", label: "Stage 2 — Segmentation", caption: "U-Net predicted tumour region (red overlay)" },
    { key: "stage3", label: "Stage 3 — Classification", caption: "EfficientNet input patch with verdict" },
  ];

  return (
    <>
      {lightbox && (
        <Lightbox
          src={lightbox.src}
          label={lightbox.label}
          caption={lightbox.caption}
          onClose={() => setLightbox(null)}
        />
      )}

      <div className="fade-up" style={{
        background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: 6, padding: "18px 20px", marginTop: 24,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
          <span style={{ fontSize: 18 }}>◈</span>
          <span style={{ color: C.white, fontWeight: 600, fontSize: 13 }}>
            Stage Visualisations
          </span>
          <Mono size={10} color={C.muted}>— click any image to expand and zoom</Mono>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
          {stages.map(({ key, label, caption }) => {
            const src = `${API}/image/${jobId}/${key}?t=${Date.now()}`;
            return (
              <div key={key} style={{
                background: C.bg, border: `1px solid ${C.border}`,
                borderRadius: 5, overflow: "hidden", cursor: "zoom-in",
              }}
                onClick={() => setLightbox({ src, label, caption })}
              >
                <div style={{
                  width: "100%", aspectRatio: "1 / 1", background: "#08111A",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  overflow: "hidden",
                }}>
                  <img
                    src={src} alt={label}
                    style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
                    onError={e => { e.currentTarget.style.display = "none"; }}
                  />
                </div>
                <div style={{ padding: "8px 10px", borderTop: `1px solid ${C.border}` }}>
                  <Mono size={10} color={C.accent}>{label}</Mono>
                  <p style={{ color: C.muted, fontSize: 10, marginTop: 2, lineHeight: 1.4 }}>{caption}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

// ── Pipeline stage definitions ────────────────────────────────────────────────

const PIPELINE_STAGES = [
  {
    key: "stage0", label: "Stage 0 — Preprocessing", icon: "◻",
    desc: "CT windowing applied: HU clipped to [-79, 304] and normalised to [0, 255]. DICOM converted to NIfTI if needed."
  },
  {
    key: "stage1", label: "Stage 1 — YOLOv8 Detection", icon: "⬡",
    desc: "Locating kidney regions across all CT slices using bounding-box detection at confidence threshold 0.10."
  },
  {
    key: "stage2", label: "Stage 2 — ResNet-UNet Segmentation", icon: "◈",
    desc: "Pixel-level tumour segmentation on the cropped kidney regions using U-Net with ResNet50 encoder."
  },
  {
    key: "stage3", label: "Stage 3 — EfficientNet Classification", icon: "◎",
    desc: "Benign / Malignant classification of detected lesions using EfficientNet-B0 fine-tuned on KiTS21 patches."
  },
];

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState("cases");
  const [jobId, setJobId] = useState(null);
  const [jobData, setJobData] = useState(null);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState(null);
  const [cancelling, setCancelling] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const pollRef = useRef(null);

  // Inject CSS + favicon once
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = GLOBAL_CSS;
    document.head.appendChild(style);

    const blob = new Blob([FAVICON_SVG], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    let link = document.querySelector("link[rel~='icon']");
    if (!link) {
      link = document.createElement("link");
      link.rel = "icon";
      document.head.appendChild(link);
    }
    link.href = url;
    document.title = "Kidney Tumour Detection System";

    return () => {
      document.head.removeChild(style);
      URL.revokeObjectURL(url);
    };
  }, []);

  // Polling — stops only when pipeline done AND SHAP not running
  const poll = useCallback(async (id) => {
    try {
      const res = await fetch(`${API}/status/${id}`);
      const data = await res.json();
      setJobData(data);

      const done = ["complete", "error", "cancelled"].includes(data.pipeline_status);
      const shapActive = data?.stages?.stage4?.status === "running";

      if (done && !shapActive) {
        clearInterval(pollRef.current);
        setPolling(false);
        setCancelling(false);
      }
    } catch {
      setError("Cannot reach backend. Make sure app.py is running on port 5000.");
      clearInterval(pollRef.current);
      setPolling(false);
    }
  }, []);

  useEffect(() => {
    if (polling && jobId) {
      pollRef.current = setInterval(() => poll(jobId), 500);
    }
    return () => clearInterval(pollRef.current);
  }, [polling, jobId, poll]);

  // Start from preloaded case
  const startFromCase = async (caseId) => {
    setError(null); setJobData(null); setJobId(null); setCancelling(false);
    try {
      const res = await fetch(`${API}/run/${caseId}`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to start case");
      setJobId(data.job_id);
      setPolling(true);
    } catch (e) {
      setError(e.message);
    }
  };

  // Start from uploaded files — uses XHR instead of fetch for upload progress events
  const startFromUpload = (files) => {
    setError(null); setJobData(null); setJobId(null);
    setCancelling(false); setUploadProgress(0);

    const form = new FormData();
    files.forEach(f => form.append("scan", f));

    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", e => {
      if (e.lengthComputable)
        setUploadProgress(Math.round((e.loaded / e.total) * 100));
    });

    xhr.addEventListener("load", () => {
      setUploadProgress(null);
      try {
        const data = JSON.parse(xhr.responseText);
        if (xhr.status >= 400) {
          setError(data.error || "Upload failed.");
        } else {
          setJobId(data.job_id);
          setPolling(true);
        }
      } catch {
        setError("Unexpected server response.");
      }
    });

    xhr.addEventListener("error", () => {
      setUploadProgress(null);
      setError("Upload failed — check your connection and try again.");
    });

    xhr.open("POST", `${API}/upload`);
    xhr.send(form);
  };

  // Cancel running pipeline
  const cancelPipeline = async () => {
    if (!jobId || cancelling) return;
    setCancelling(true);
    try {
      await fetch(`${API}/cancel/${jobId}`, { method: "POST" });
    } catch {
      setCancelling(false);
    }
  };

  // SHAP trigger
  const triggerShap = async () => {
    if (!jobId) return;
    try {
      await fetch(`${API}/shap/${jobId}`, { method: "POST" });
      setPolling(true);
    } catch {
      setError("SHAP request failed.");
    }
  };

  const pipelineStatus = jobData?.pipeline_status;
  const isQueued = pipelineStatus === "queued" || (!pipelineStatus && polling);
  const isRunning = pipelineStatus === "running";
  const isComplete = pipelineStatus === "complete";
  const isError = pipelineStatus === "error";
  const isCancelled = pipelineStatus === "cancelled";
  const isActive = isQueued || isRunning;
  const shapStatus = jobData?.stages?.stage4?.status;

  const reset = () => {
    setJobId(null); setJobData(null);
    setPolling(false); setError(null); setCancelling(false);
    clearInterval(pollRef.current);
  };

  return (
    <>
      <div className="scanline-overlay" />

      <div style={{ maxWidth: 820, margin: "0 auto", padding: "32px 24px" }}>

        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <div style={{
            display: "flex", alignItems: "flex-start",
            justifyContent: "space-between", gap: 16, flexWrap: "wrap",
          }}>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                <div style={{
                  width: 6, height: 32, background: C.accent,
                  boxShadow: `0 0 12px ${C.accent}88`
                }} />
                <h1 style={{
                  color: C.white, fontSize: 20, fontWeight: 700,
                  letterSpacing: "-0.02em", lineHeight: 1.2
                }}>
                  Kidney Tumour Detection System
                </h1>
              </div>
              <Mono size={11} color={C.muted}>
                YOLOv8 · ResNet-UNet · EfficientNet-B0 · KiTS21 · PROOF-OF-CONCEPT
              </Mono>
            </div>
            <div style={{ textAlign: "right" }}>
              <Tag
                label={
                  isQueued ? "queued" :
                    isRunning ? "processing" :
                      isComplete ? "complete" :
                        isError ? "error" :
                          isCancelled ? "cancelled" : "idle"
                }
                color={
                  isRunning ? C.warn :
                    isComplete ? C.accent :
                      isError ? C.danger :
                        isCancelled ? C.muted : C.muted
                }
              />
              {jobId && <div style={{ marginTop: 4 }}><Mono>JOB {jobId}</Mono></div>}
            </div>
          </div>
          <div style={{
            marginTop: 16, height: 1,
            background: `linear-gradient(90deg, ${C.accent}44, transparent)`
          }} />
        </div>

        {/* Global error banner */}
        {error && (
          <div style={{ marginBottom: 16 }}>
            <Alert message={error} />
          </div>
        )}

        {/* Backend error surfaced clearly */}
        {isError && jobData?.error && (
          <div style={{ marginBottom: 16 }}>
            <Alert message={`Pipeline error: ${jobData.error}`} />
          </div>
        )}

        {/* Input panel — shown only before pipeline starts */}
        {!isActive && !isComplete && !isError && !isCancelled && (
          <div className="fade-up" style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 8, overflow: "hidden", marginBottom: 24,
          }}>
            {/* Tabs */}
            <div style={{ display: "flex", borderBottom: `1px solid ${C.border}` }}>
              {[
                { id: "cases", label: "Choose Test Case" },
                { id: "upload", label: "Upload Your Own" },
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    flex: 1, padding: "14px", background: "transparent",
                    border: "none",
                    borderBottom: activeTab === tab.id
                      ? `2px solid ${C.accent}` : "2px solid transparent",
                    color: activeTab === tab.id ? C.accent : C.muted,
                    fontFamily: "'DM Mono', monospace",
                    fontSize: 12, letterSpacing: "0.06em",
                    cursor: "pointer", textTransform: "uppercase",
                    transition: "all 0.2s",
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div style={{ padding: "20px" }}>
              {activeTab === "cases"
                ? <CasePicker onSelect={startFromCase} disabled={isActive} />
                : <UploadZone onStart={startFromUpload} disabled={isActive} uploadProgress={uploadProgress} />
              }
            </div>
          </div>
        )}

        {/* Queued state — shown immediately so screen never looks frozen */}
        {isQueued && !isRunning && (
          <div className="fade-up" style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 6, padding: "16px 20px", marginBottom: 16,
            display: "flex", alignItems: "center", gap: 12,
          }}>
            <Spinner size={14} />
            <Mono color={C.muted}>Pipeline queued — starting shortly…</Mono>
          </div>
        )}

        {/* Stage cards */}
        {(isActive || isComplete || isError) && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 24 }}>

            {/* Live indicator + cancel button */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              {isRunning && <Spinner size={12} />}
              <span style={{ color: C.muted, fontSize: 12, flex: 1 }}>
                {isRunning ? "Pipeline running…" :
                  isComplete ? "All stages complete" :
                    isError ? "Pipeline stopped with error" : ""}
              </span>
              {isRunning && (
                <>
                  <Mono color={C.warn}>LIVE</Mono>
                  <button
                    onClick={cancelPipeline}
                    disabled={cancelling}
                    style={{
                      padding: "5px 14px",
                      background: "transparent",
                      border: `1px solid ${C.danger}66`,
                      color: cancelling ? C.muted : C.danger,
                      borderRadius: 3,
                      fontFamily: "'DM Mono', monospace",
                      fontSize: 11, letterSpacing: "0.06em",
                      cursor: cancelling ? "not-allowed" : "pointer",
                      textTransform: "uppercase",
                      transition: "all 0.2s",
                    }}
                  >
                    {cancelling ? "cancelling…" : "cancel"}
                  </button>
                </>
              )}
            </div>

            {PIPELINE_STAGES.map((s, i) => (
              <StageCard
                key={s.key} stage={s}
                data={jobData?.stages?.[s.key]}
                index={i}
              />
            ))}
          </div>
        )}

        {/* Cancelled message */}
        {isCancelled && (
          <div style={{ marginBottom: 24 }}>
            <Alert
              message="Pipeline was cancelled. You can start a new analysis below."
              color={C.muted}
            />
          </div>
        )}

        {/* Image gallery — shown after summary, before SHAP */}
        {isComplete && (
          <ImageGallery jobId={jobId} isComplete={isComplete} />
        )}

        {/* SHAP card */}
        {isComplete && (
          <div className="fade-up" style={{
            background: C.surface,
            border: `1px solid ${shapStatus === "complete" ? C.accent + "55" : C.border}`,
            borderRadius: 6, padding: "16px 20px", marginBottom: 24,
          }}>
            <div style={{
              display: "flex", alignItems: "center",
              justifyContent: "space-between", gap: 12, flexWrap: "wrap",
            }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <span style={{ fontSize: 18 }}>◇</span>
                  <span style={{ color: C.white, fontWeight: 600, fontSize: 13 }}>
                    Stage 4 — SHAP Explainability
                  </span>
                  {shapStatus === "complete" && (
                    <>
                      <DurationBadge seconds={jobData?.stages?.stage4?.duration_s} />
                      <Tag label="complete" color={C.accent} />
                    </>
                  )}
                  {shapStatus === "running" && <Tag label="running" color={C.warn} />}
                </div>
                <p style={{ color: C.muted, fontSize: 12, lineHeight: 1.6 }}>
                  Optional. Generates attribution heatmaps showing which image regions
                  drove each model decision. Takes ~8–30 seconds.
                </p>
              </div>
              {!shapStatus && (
                <button
                  onClick={triggerShap}
                  style={{
                    padding: "10px 20px", background: "transparent",
                    border: `1px solid ${C.accent}`, color: C.accent,
                    borderRadius: 4, fontFamily: "'DM Mono', monospace",
                    fontSize: 12, letterSpacing: "0.06em",
                    cursor: "pointer", textTransform: "uppercase",
                    whiteSpace: "nowrap", transition: "all 0.2s",
                  }}
                >
                  RUN SHAP ANALYSIS
                </button>
              )}
            </div>

            {shapStatus === "running" && (
              <div style={{ marginTop: 12 }}>
                <ProgressBar value={jobData?.stages?.stage4?.progress || 0} color={C.warn} />
                <div style={{ marginTop: 6 }}>
                  <Mono color={C.warn}>{jobData?.stages?.stage4?.message}</Mono>
                </div>
              </div>
            )}
            {shapStatus === "complete" && jobData?.stages?.stage4?.result && (
              <ShapAnalysis
                jobId={jobId}
                result={jobData.stages.stage4.result}
                shapStatus={shapStatus}
              />
            )}
          </div>
        )}

        {/* Summary */}
        {isComplete && jobData?.summary && (
          <SummaryPanel summary={jobData.summary} />
        )}

        {/* Reset */}
        {(isComplete || isError || isCancelled) && (
          <div style={{ marginTop: 20, textAlign: "center" }}>
            <button
              onClick={reset}
              style={{
                background: "transparent", border: `1px solid ${C.border}`,
                color: C.muted, padding: "10px 28px", borderRadius: 4,
                fontFamily: "'DM Mono', monospace", fontSize: 12,
                cursor: "pointer", letterSpacing: "0.06em",
                textTransform: "uppercase", transition: "all 0.2s",
              }}
            >
              ANALYSE NEW CASE
            </button>
          </div>
        )}

        {/* Footer */}
        <div style={{
          marginTop: 48, paddingTop: 16,
          borderTop: `1px solid ${C.border}`,
          display: "flex", justifyContent: "space-between",
          flexWrap: "wrap", gap: 8,
        }}>
          <Mono size={10} color={C.muted}>
            Daniel Osaseri Okundaye · MIVA Open University · B.Sc. CS 2026
          </Mono>
          <Mono size={10} color={C.muted}>
            RESEARCH PROTOTYPE · KiTS21 DATASET · NOT FOR CLINICAL USE
          </Mono>
        </div>

      </div>
    </>
  );
}