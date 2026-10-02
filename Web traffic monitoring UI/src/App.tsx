import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'

type Frame = { id: string; frame_index: number; timestamp_sec: number; analyzed: boolean; analysis?: any; yolo?: any; image_url: string; error?: string }
type Session = { id: string; camera_id?: number | null; filename: string; media_type: string; created_at: string; duration?: number; width?: number; height?: number; provider?: string; model?: string; processing_status?: string; processing_error?: string; llm_analysis?: any; llm_updated_at?: string; llm_status?: string; frames: Frame[]; analyzed_count?: number; frame_count?: number; latest_analysis?: any }
type Camera = { id: number; name: string; location: string; status: string; source?: string | number | null }
type LibraryMedia = { name: string; size_bytes: number; extension: string }
type Settings = { provider: string; ollama_model: string; gemini_model: string; gpt_model: string; detector_model: string; sample_every_seconds: number; llm_sample_every_seconds: number; detector_fps: number; max_frames: number; motorcycle_alert_threshold: number; api_keys?: { gemini_configured: boolean; gpt_configured: boolean } }
const API = '/api'
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try { const body = await response.json(); detail = body.detail || detail } catch { /* use HTTP status */ }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}
const panel: CSSProperties = { background: '#0d1625', border: '1px solid #1e2d42', borderRadius: 10 }
const subtle: CSSProperties = { color: '#7f8da3' }
const button: CSSProperties = { background: '#14233a', border: '1px solid #29415f', color: '#dbeafe', borderRadius: 7, padding: '8px 11px', cursor: 'pointer' }
function formatTime(value?: number | string) { if (value == null) return '—'; const n = typeof value === 'string' ? Date.parse(value) / 1000 : value; if (!Number.isFinite(n)) return String(value); return new Date(n * 1000).toISOString().slice(11, 19) }
function download(name: string, text: string, type: string) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); URL.revokeObjectURL(a.href) }

export default function App() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [library, setLibrary] = useState<LibraryMedia[]>([])
  const [libraryName, setLibraryName] = useState('')
  const [cameraId, setCameraId] = useState(1)
  const [session, setSession] = useState<Session | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [history, setHistory] = useState<Session[]>([])
  const [logs, setLogs] = useState<any[]>([])
  const [settings, setSettings] = useState<Settings>({ provider: 'ollama', ollama_model: 'qwen2.5vl:7b', gemini_model: 'gemini-2.5-flash', gpt_model: 'gpt-4o-mini', detector_model: 'yolo11s.pt', sample_every_seconds: 5, llm_sample_every_seconds: 5, detector_fps: 5, max_frames: 30000, motorcycle_alert_threshold: 150 })
  const [tab, setTab] = useState<'analysis' | 'history' | 'settings'>('analysis')
  const [showLogs, setShowLogs] = useState(false)
  const [busy, setBusy] = useState(false)
  const [apiOnline, setApiOnline] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const frame = session?.frames?.[frameIndex]
  const result = session?.llm_analysis || frame?.analysis
  const yolo = frame?.yolo
  const traffic = result?.traffic || {}
  const flood = result?.flood || result?.flooding || {}
  const vehicles = traffic?.vehicles || result?.vehicles || {}
  const weather = result?.weather_vi || result?.weather?.condition || result?.weather || 'Chưa phân tích'
  const alertText = typeof result?.safety_alert === 'string' ? result.safety_alert : result?.safety_alert?.message
  const selectedCamera = cameras.find(c => c.id === cameraId)
  const model = settings.provider === 'gpt' ? settings.gpt_model : settings.provider === 'gemini' ? settings.gemini_model : settings.ollama_model
  const totalVehicles = [vehicles.motorcycle, vehicles.car, vehicles.bus_truck].reduce((a: number, b: any) => a + (Number(b) || 0), 0)

  async function refresh() {
    try {
      const [cams, hist, logRows, config, media] = await Promise.all([
        request<Camera[]>('/cameras'), request<Session[]>('/history?limit=200'), request<any[]>('/logs'), request<Settings>('/settings'), request<LibraryMedia[]>('/media'),
      ])
      setCameras(cams); setHistory(hist); setLogs(logRows); setSettings(s => ({ ...s, ...config })); setLibrary(media)
      setApiOnline(true)
      setLibraryName(current => current || media[0]?.name || '')
      if (cams.length && !cams.some(c => c.id === cameraId)) setCameraId(cams[0].id)
    } catch (e) { setApiOnline(false); setError((e as Error).message) }
  }
  useEffect(() => { void refresh() }, [])
  useEffect(() => { if (session) setFrameIndex(i => Math.min(i, Math.max(session.frames.length - 1, 0))) }, [session])
  useEffect(() => {
    if (!session || (!['queued', 'vision_processing', 'llm_processing'].includes(session.processing_status || '') && !['queued', 'processing'].includes(session.llm_status || '') && !(session.media_type === 'image' && !session.llm_analysis))) return
    let active = true
    const timer = window.setInterval(async () => {
      try {
        const latest = await request<Session>('/sessions/' + session.id)
        if (!active) return
        setSession(latest)
        if (latest.processing_status === 'vision_processing') setNotice('Video đang được detector xử lý tuần tự; có thể phát video ngay. LLM sẽ phân tích theo khoảng thời gian đã cài đặt.')
        else if (latest.processing_status === 'llm_processing') setNotice('Detector đã xong; LLM đang phân tích các frame đã lấy mẫu.')
        if (latest.llm_status === 'complete' && latest.llm_analysis) { setNotice('Đã cập nhật kết quả Vision AI.'); await refresh() }
        else if (latest.processing_status === 'failed') { setError(latest.processing_error || 'Xử lý video thất bại.'); setNotice('') }
      } catch { /* keep polling through transient request errors */ }
    }, 2000)
    return () => { active = false; window.clearInterval(timer) }
  }, [session?.id, session?.processing_status, session?.llm_status])

  async function upload(file: File) {
    setBusy(true); setError(''); setNotice('Đang tải ảnh/video lên...')
    try {
      const data = new FormData(); data.append('file', file); data.append('camera_id', String(cameraId))
      const created = await request<Session>('/sessions', { method: 'POST', body: data })
      setSession(created); setFrameIndex(0)
      if (created.media_type === 'image' && created.frames[0]) {
        setNotice(`Đã tải ảnh lên. Đang phân tích bằng ${model}...`)
        const analysisData = new FormData()
        analysisData.append('provider', settings.provider); analysisData.append('model', model); analysisData.append('frame_id', created.frames[0].id)
        const response = await request<any>(`/sessions/${created.id}/analyze`, { method: 'POST', body: analysisData })
        setSession(response.session); setFrameIndex(0)
        setNotice(`Đã nhận ảnh. Vision AI đang phân tích bằng ${response.provider}/${response.model}...`)
      } else setNotice(`Đã tải ${created.filename}. Đang xử lý nền ở tối đa ${settings.detector_fps} FPS; LLM phân tích mỗi ${settings.llm_sample_every_seconds} giây. Bạn có thể phát video ngay.`)
      await refresh()
    } catch (e) { setError((e as Error).message); setNotice('') }
    finally { setBusy(false) }
  }  async function loadLibraryVideo() {
    if (!libraryName) { setError('Không có video trong thư mục data.'); return }
    setBusy(true); setError(''); setNotice('Đang đọc video trong thư mục data và trích xuất frame...')
    try {
      const created = await request<Session>('/sessions/from-library', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: libraryName, camera_id: cameraId, sample_every_seconds: 1 / settings.detector_fps }) })
      setSession(created); setFrameIndex(0); setNotice(`Đ? m? ${created.filename}, trích ${created.frames.length} frame.`); await refresh()
    } catch (e) { setError((e as Error).message); setNotice('') }
    finally { setBusy(false) }
  }
  async function analyze(all: boolean) {
    if (!session) { setError('Hãy tải ảnh hoặc video lên trước.'); return }
    const targets = all ? session.frames : frame ? [frame] : []
    if (!targets.length) { setError('Phiên này không có frame để phân tích.'); return }
    setBusy(true); setError('')
    try {
      setNotice(all ? `Đã đưa ${targets.length} frame vào hàng đợi Vision AI.` : 'Đang phân tích frame bằng Vision AI...')
      const data = new FormData(); data.append('provider', settings.provider); data.append('model', model)
      if (!all && frame) data.append('frame_id', frame.id)
      const response = await request<any>('/sessions/' + session.id + '/analyze', { method: 'POST', body: data })
      setSession(response.session)
      await refresh()
    } catch (e) { setError((e as Error).message); setNotice('') }
    finally { setBusy(false) }
  }
  async function capture() {
    setBusy(true); setError('')
    try {
      const response = await fetch(`${API}/cameras/${cameraId}/capture`)
      if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || 'Không chụp được camera') }
      const file = new File([await response.blob()], `camera-${cameraId}-${Date.now()}.jpg`, { type: 'image/jpeg' }); await upload(file)
    } catch (e) { setError((e as Error).message); setBusy(false) }
  }
  async function openHistory(id: string) {
    setBusy(true); setError('')
    try {
      const detail = await request<Session>('/sessions/' + id)
      setSession(detail); setFrameIndex(0); setTab('analysis')
      if (detail.camera_id != null) setCameraId(detail.camera_id)
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  async function selectCamera(id: number) {
    setCameraId(id); setError(''); setNotice(''); setTab('analysis')
    const latest = history.find(item => item.camera_id === id)
    if (latest) await openHistory(latest.id)
    else { setSession(null); setFrameIndex(0) }
  }
  async function saveSettings(next: Settings) {
    setBusy(true); setError('')
    try { const updated = await request<Settings>('/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(next) }); setSettings(s => ({ ...s, ...updated })); setNotice('Đã lưu cài đặt.') }
    catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  async function sendAlert() {
    const message = alertText || `Cảnh báo ùn tắc tại ${selectedCamera?.name || 'camera'}: ${congestion}`
    try { await request('/alerts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: session?.id, frame_id: frame?.id, message }) }); setNotice('Đã ghi cảnh báo vào system logs.'); await refresh() }
    catch (e) { setError((e as Error).message) }
  }
  const jsonResult = useMemo(() => result ? JSON.stringify(result, null, 2) : '{}', [result])

  return <div style={{ minHeight: '100vh', background: '#080b10', color: '#dbe4f0', fontFamily: 'Inter, system-ui, sans-serif' }}>
    <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '13px 20px', background: '#0a0e18', borderBottom: '1px solid #1e2d42' }}>
      <div><strong style={{ color: 'white' }}>TrafficVision AI</strong><div style={{ ...subtle, fontSize: 12, marginTop: 4 }}>Hệ thống giám sát giao thông thông minh</div></div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}><span style={{ color: apiOnline ? '#4ade80' : '#f87171', fontSize: 12 }}>● API {apiOnline ? 'connected' : 'offline'}</span><span style={{ ...button, cursor: 'default', fontFamily: 'monospace' }}>{model}</span><button style={button} onClick={() => { setShowLogs(v => !v); void refresh() }}>System Logs</button></div>
    </header>
    {showLogs && <div style={{ padding: 12, maxHeight: 150, overflow: 'auto', background: '#09101c', borderBottom: '1px solid #1e2d42' }}>{logs.map((l, i) => <div key={l.id || i} style={{ fontSize: 12, padding: 3 }}><span style={subtle}>{formatTime(l.created_at)} </span><b style={{ color: l.level === 'WARN' || l.level === 'ERROR' ? '#fbbf24' : '#4ade80' }}>{l.level}</b>?{l.message}</div>)}</div>}
    {(error || notice) && <div style={{ margin: '12px 18px 0', padding: 10, borderRadius: 7, background: error ? '#3b1720' : '#10251f', color: error ? '#fca5a5' : '#86efac' }}>{error || notice}<button style={{ float: 'right', background: 'transparent', border: 0, color: 'inherit', cursor: 'pointer' }} onClick={() => { setError(''); setNotice('') }}>×</button></div>}
    <div style={{ display: 'grid', gridTemplateColumns: '220px minmax(320px,1fr) minmax(340px,420px)', minHeight: 'calc(100vh - 64px)' }}>
      <aside style={{ background: '#09101c', borderRight: '1px solid #1e2d42', padding: 12 }}><b style={{ color: '#718096', fontSize: 11 }}>DANH SÁCH CAMERA</b>
        {cameras.map(cam => <button key={cam.id} disabled={busy} onClick={() => void selectCamera(cam.id)} style={{ display: 'block', textAlign: 'left', width: '100%', marginTop: 8, padding: 10, background: cameraId === cam.id ? '#102139' : 'transparent', color: '#cbd5e1', border: cameraId === cam.id ? '1px solid #245084' : '1px solid transparent', borderRadius: 8, cursor: busy ? 'wait' : 'pointer' }}><div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}><b>CAM_{String(cam.id).padStart(3, '0')}</b><span style={{ color: cam.status === 'congested' ? '#f87171' : '#4ade80' }}>{cam.status === 'congested' ? 'TẮC' : 'THÔNG'}</span></div><div style={{ fontSize: 11, marginTop: 5 }}>{cam.name}</div><div style={{ ...subtle, fontSize: 10, marginTop: 3 }}>{cam.location}</div></button>)}
        <div style={{ ...panel, padding: 10, marginTop: 18, fontSize: 12 }}>Tổng camera <b style={{ float: 'right' }}>{cameras.length}</b><br/><span style={subtle}>Phiên phân tích</span><b style={{ float: 'right' }}>{history.length}</b></div>
      </aside>
      <main style={{ padding: 14, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ ...panel, display: 'flex', justifyContent: 'space-between', padding: 10, fontSize: 12 }}><span><b>{frame ? `frame_${String(frame.frame_index).padStart(4, '0')}` : 'Chưa có frame'}</b>?{session?.filename || selectedCamera?.name}</span><span style={subtle}>{frame ? `${frame.timestamp_sec.toFixed(2)}s` : ''}{session?.width ? `?${session.width}×${session.height}` : ''}</span></div>
        <div style={{ ...panel, flex: 1, minHeight: 300, display: 'grid', placeItems: 'center', padding: 10, background: '#050810' }}>{session?.media_type === 'video' ? <video key={session.id} src={`${API}/sessions/${session.id}/media`} controls preload="metadata" style={{ width: '100%', maxHeight: '66vh', objectFit: 'contain' }}/> : frame ? <img src={frame.image_url} alt="Analyzed frame" style={{ width: '100%', maxHeight: '66vh', objectFit: 'contain' }}/> : <div style={{ textAlign: 'center', ...subtle }}><div style={{ fontSize: 40, marginBottom: 8 }}>▣</div>Chọn ảnh/video để bắt đầu phân tích</div>}</div>
        <div style={{ ...panel, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: 9 }}><button style={button} disabled={!frameIndex} onClick={() => setFrameIndex(i => Math.max(0, i - 1))}>‹ Frame trước</button><span style={{ ...subtle, fontSize: 12 }}>Frame {frame ? frameIndex + 1 : 0} / {session?.frames.length || 0}</span><button style={button} disabled={!session || frameIndex >= session.frames.length - 1} onClick={() => setFrameIndex(i => Math.min((session?.frames.length || 1) - 1, i + 1))}>Frame sau ›</button></div>
        <div style={{ ...panel, padding: 10 }}><div style={{ ...subtle, fontSize: 11, marginBottom: 8 }}>DẢI KHUNG HÌNH ({session?.frames.length || 0})</div><div style={{ display: 'flex', gap: 7, overflowX: 'auto' }}>{session?.frames.map((f, i) => <button key={f.id} onClick={() => setFrameIndex(i)} style={{ padding: 2, border: i === frameIndex ? '2px solid #3b82f6' : '2px solid transparent', borderRadius: 5, background: '#0f1828', cursor: 'pointer' }}><img src={f.image_url} style={{ width: 85, height: 52, objectFit: 'cover', display: 'block' }}/><span style={{ fontSize: 10, color: '#94a3b8' }}>{f.timestamp_sec}s</span></button>)}</div></div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}><button style={button} disabled={busy} onClick={() => void analyze(false)}>✨ Phân tích frame</button><button style={button} disabled={busy} onClick={() => void analyze(true)}>▣ Phân tích toàn bộ video</button><button style={button} disabled={busy} onClick={() => fileRef.current?.click()}>Tải ảnh/video lên</button><input ref={fileRef} type="file" accept="image/*,video/*" hidden onChange={e => { const f = e.target.files?.[0]; if (f) void upload(f); e.currentTarget.value = '' }}/><select value={libraryName} onChange={e => setLibraryName(e.target.value)} disabled={!library.length || busy} style={{ maxWidth: 210, ...button }}>{library.length ? library.map(media => <option key={media.name} value={media.name}>{media.name}</option>) : <option>Không có video trong data/</option>}</select><button style={button} disabled={!library.length || busy} onClick={() => void loadLibraryVideo()}>Mở video data/</button><button style={button} disabled={busy} onClick={() => void capture()}>Chụp từ camera</button><button style={button} disabled={!session} onClick={() => session && download(`${session.id}.json`, JSON.stringify(session, null, 2), 'application/json')}>Xuất JSON</button><button style={button} disabled={!session} onClick={() => session && download(`${session.id}.csv`, ['frame_id,timestamp_sec,congestion,flooded,motorcycles,cars,bus_truck,weather,description', ...session.frames.map(f => { const a = f.analysis || {}; const t = a.traffic || {}; const v = t.vehicles || {}; return [f.id, f.timestamp_sec, t.congestion_level_vi || t.congestion_level || '', (a.flood || {}).is_flooded ?? '', v.motorcycle ?? '', v.car ?? '', v.bus_truck ?? '', a.weather_vi || a.weather || '', JSON.stringify(a.ai_description || '')].join(',') })].join('\n'), 'text/csv')}>Xuất CSV</button></div>
      </main>
      <aside style={{ background: '#090f1a', borderLeft: '1px solid #1e2d42', padding: 12, minWidth: 0 }}>
        <nav style={{ display: 'flex', borderBottom: '1px solid #1e2d42', marginBottom: 12 }}>{(['analysis', 'history', 'settings'] as const).map(t => <button key={t} onClick={() => setTab(t)} style={{ flex: 1, padding: 10, background: 'transparent', border: 0, borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent', color: tab === t ? '#60a5fa' : '#718096', cursor: 'pointer' }}>{t === 'analysis' ? 'Phân tích' : t === 'history' ? 'Lịch sử' : 'Cài đặt'}</button>)}</nav>
        {tab === 'analysis' && <div style={{ display: 'grid', gap: 10 }}><div style={{ ...panel, padding: 13 }}><div style={subtle}>{yolo?.model || settings.detector_model} · NHÃN GIAO THÔNG</div><b style={{ display: 'block', marginTop: 8, color: yolo?.label === 'Ùn tắc' ? '#f87171' : yolo?.label === 'Đông đúc' ? '#fbbf24' : '#4ade80' }}>{yolo?.label || 'ĐANG CHỜ DETECTOR'}</b><div style={{ ...subtle, marginTop: 7, fontSize: 11 }}>Nhãn được tính từ kết quả nhận diện và theo dõi phương tiện.</div></div>
          <div style={{ ...panel, padding: 13 }}><div style={subtle}>TÌNH TRẠNG NGẬP LỤT</div><b style={{ display: 'block', marginTop: 8, color: flood.is_flooded ? '#fbbf24' : '#4ade80' }}>{flood.status_vi || (flood.is_flooded ? 'CÓ NGẬP' : result ? 'KHÔ RÁO' : 'CHƯA PHÂN TÍCH')}</b><div style={{ ...subtle, marginTop: 7, fontSize: 12 }}>Diện tích ngập: {flood.flood_area_percent ?? flood.area_percent_estimate ?? '—'}{result ? '%' : ''}</div></div>
          <div style={{ ...panel, padding: 12, display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 4, textAlign: 'center' }}><div><b>{vehicles.motorcycle ?? '—'}</b><div style={{ ...subtle, fontSize: 10 }}>Xe máy (LLM)</div></div><div><b>{vehicles.car ?? '—'}</b><div style={{ ...subtle, fontSize: 10 }}>Ô tô (LLM)</div></div><div><b>{vehicles.bus_truck ?? '—'}</b><div style={{ ...subtle, fontSize: 10 }}>Bus/tải (LLM)</div></div><div><b style={{ fontSize: 12 }}>{weather}</b><div style={{ ...subtle, fontSize: 10 }}>Thời tiết</div></div></div>
          {result && <div style={{ ...panel, padding: 12, borderColor: alertText ? '#783641' : '#1e2d42' }}><b style={{ color: '#f87171', fontSize: 12 }}>⚠ CẢNH BÁO AN TOÀN</b><p style={{ color: '#fca5a5', fontSize: 12, lineHeight: 1.5 }}>{alertText || 'Không có cảnh báo.'}</p></div>}
          <div style={{ ...panel, padding: 12 }}><b style={{ fontSize: 12 }}>Nhận định chi tiết từ Vision AI</b><p style={{ color: '#a9b6c9', fontSize: 12, lineHeight: 1.55 }}>{result?.ai_description || (session?.llm_status === 'processing' || session?.llm_status === 'queued' ? 'Vision AI đang phân tích frame đầu tiên...' : 'Tải video/ảnh và phân tích để xem nhận định.')}</p><div style={{ ...subtle, fontSize: 10 }}>{session?.llm_updated_at ? `Cập nhật ${formatTime(session.llm_updated_at)}` : ''}</div></div>
          <details style={{ ...panel, padding: 10 }}><summary style={{ cursor: 'pointer', fontSize: 12 }}>Dữ liệu JSON Output</summary><div style={{ display: 'flex', gap: 6, marginTop: 8 }}><button style={button} onClick={() => void navigator.clipboard.writeText(jsonResult)}>Copy JSON</button><button style={button} onClick={() => download(`${frame?.id || 'analysis'}.json`, jsonResult, 'application/json')}>Tải JSON</button></div><pre style={{ whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: 230, color: '#64d8a3', fontSize: 10 }}>{jsonResult}</pre></details><button style={{ ...button, background: '#3b1720', color: '#fca5a5' }} disabled={!result} onClick={() => void sendAlert()}>Gửi cảnh báo</button></div>}
        {tab === 'history' && <div style={{ display: 'grid', gap: 8 }}>{history.length === 0 && <p style={subtle}>Chưa có lịch sử phân tích</p>}{history.map(h => <button key={h.id} onClick={() => void openHistory(h.id)} style={{ ...panel, textAlign: 'left', padding: 11, color: '#dbe4f0', cursor: 'pointer' }}><b>{h.filename}</b><div style={{ ...subtle, fontSize: 11, marginTop: 5 }}>{formatTime(h.created_at)} · {h.analyzed_count || 0}/{h.frame_count || 0} frame đã phân tích</div></button>)}</div>}
        {tab === 'settings' && <div style={{ display: 'grid', gap: 12 }}><label style={{ ...panel, padding: 12, fontSize: 12 }}>Bộ nhận diện<select value={settings.detector_model} onChange={e => setSettings(s => ({ ...s, detector_model: e.target.value }))} style={{ display: 'block', width: '100%', marginTop: 7, padding: 8, background: '#0f1828', color: '#dbeafe', border: '1px solid #29415f', borderRadius: 5 }}><option value="yolo11s.pt">YOLO11s</option><option value="rtdetr-l.pt">RT-DETR-L</option></select><small style={subtle}>Detector xử lý tuần tự tối đa 30 FPS; LLM ước lượng số xe.</small></label><label style={panel as any}><span style={subtle}>Model Vision</span><select value={settings.provider} onChange={e => setSettings(s => ({ ...s, provider: e.target.value }))} style={{ display: 'block', width: '100%', marginTop: 7, padding: 8, background: '#0f1828', color: '#dbeafe', border: '1px solid #29415f', borderRadius: 5 }}><option value="ollama">Local Ollama (Qwen2.5-VL)</option><option value="gemini">Google Gemini</option><option value="gpt">OpenAI GPT</option></select><input value={model} onChange={e => setSettings(s => s.provider === 'gpt' ? { ...s, gpt_model: e.target.value } : s.provider === 'gemini' ? { ...s, gemini_model: e.target.value } : { ...s, ollama_model: e.target.value })} style={{ width: '100%', marginTop: 7, padding: 8, background: '#0f1828', color: '#dbeafe', border: '1px solid #29415f', borderRadius: 5 }}/><small style={subtle}>{settings.provider === 'ollama' ? 'Chạy local, không cần API key. Cần Ollama đang chạy và model đã tải.' : `API key: ${settings.provider === 'gpt' ? (settings.api_keys?.gpt_configured ? 'đã cấu hình' : 'chưa cấu hình') : (settings.api_keys?.gemini_configured ? 'đã cấu hình' : 'chưa cấu hình')}`}</small></label>
          <label style={{ ...panel, padding: 12, fontSize: 12 }}>LLM tự phân tích mỗi (giây)<input type="number" min="0.25" step="0.25" value={settings.llm_sample_every_seconds} onChange={e => setSettings(s => ({ ...s, llm_sample_every_seconds: Number(e.target.value) }))} style={{ width: '100%', marginTop: 7, padding: 7, background: '#0f1828', color: 'white', border: '1px solid #29415f' }}/></label>
          <label style={{ ...panel, padding: 12, fontSize: 12 }}>Ngưỡng cảnh báo xe máy<input type="number" min="1" value={settings.motorcycle_alert_threshold} onChange={e => setSettings(s => ({ ...s, motorcycle_alert_threshold: Number(e.target.value) }))} style={{ width: '100%', marginTop: 7, padding: 7, background: '#0f1828', color: 'white', border: '1px solid #29415f' }}/></label><button style={button} disabled={busy} onClick={() => void saveSettings(settings)}>Lưu cài đặt</button><p style={{ ...subtle, fontSize: 11, lineHeight: 1.5 }}>{settings.provider === 'ollama' ? 'Ảnh được xử lý bởi Ollama đang chạy trên máy local.' : 'API key được backend đọc từ biến môi trường, không gửi hoặc lưu trong giao diện web.'}</p></div>}
      </aside>
    </div>
  </div>
}

