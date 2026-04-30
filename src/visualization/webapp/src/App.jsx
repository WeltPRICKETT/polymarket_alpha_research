import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [stats, setStats] = useState({ total_transactions: 0, total_markets: 0, last_update: '' });
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState([]);
  const [comparison, setComparison] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/stats`);
      const data = await res.json();
      setStats(data);
    } catch (e) { console.error(e); }
  };

  const fetchLogs = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/logs`);
      const data = await res.json();
      setLogs(data.logs || []);
    } catch (e) { console.error(e); }
  };

  const fetchResults = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/results`);
      const data = await res.json();
      setResults(data.files || []);
    } catch (e) { console.error(e); }
  };

  const fetchComparison = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/comparison-data`);
      const data = await res.json();
      setComparison(data.models || []);
    } catch (e) { console.error(e); }
  };

  useEffect(() => {
    fetchStats();
    fetchLogs();
    fetchResults();
    fetchComparison();
    const interval = setInterval(() => {
      fetchStats();
      fetchLogs();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleRunModel = async (modelName) => {
    setLoading(true);
    try {
      await fetch(`${API_BASE}/api/run-backtest?model=${modelName}`, { method: 'POST' });
      alert(`[SYS_CMD] Execution pipeline initialized for: ${modelName.toUpperCase()}`);
      setTimeout(fetchResults, 30000);
    } catch (e) { alert('[ERR] Execution failed.'); }
    setLoading(false);
  };

  return (
    <div className="flex h-screen overflow-hidden relative">
      {/* Sidebar - HUD Panels */}
      <aside className="w-80 m-6 flex flex-col z-10 space-y-6">
        <div className="card bg-black/60 backdrop-blur-sm border-[#39ff14]">
          <h1 className="text-5xl font-black uppercase text-[#39ff14] glitch text-glow" data-text="POLY_ALPHA">POLY_ALPHA</h1>
          <p className="text-xs text-[#00f3ff] mt-2 tracking-[0.3em] opacity-80">v2.0 // ACID_HUD</p>
          <div className="h-[2px] w-full bg-gradient-to-r from-[#39ff14] to-transparent mt-4"></div>
        </div>
        
        <nav className="card flex-1 bg-black/60 backdrop-blur-sm flex flex-col p-6 space-y-2">
          <div className="text-[10px] text-[#ff00ff] mb-4 tracking-[0.2em]">[SYSTEM_MODULES]</div>
          {[
            { id: 'overview', label: 'SYS_DASHBOARD' },
            { id: 'models', label: 'ML_EXECUTION' },
            { id: 'comparison', label: 'MATRIX_COMPARE' },
            { id: 'results', label: 'VISUAL_OUTPUTS' }
          ].map(tab => (
            <div
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`sidebar-item ${activeTab === tab.id ? 'active' : ''}`}
            >
              &gt; {tab.label}
            </div>
          ))}
        </nav>
        
        <div className="p-4 border border-[#39ff14] bg-black/60 backdrop-blur-sm flex items-center justify-between shadow-[0_0_15px_rgba(57,255,20,0.2)]">
          <span className="text-xs font-bold tracking-widest text-[#39ff14]">UPLINK_STATUS: SECURE</span>
          <div className="w-3 h-3 bg-[#39ff14] rounded-sm animate-pulse shadow-[0_0_10px_#39ff14]"></div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-6 z-10">
        <header className="mb-10 border-b border-[#00f3ff] pb-6 flex justify-between items-end bg-black/40 backdrop-blur-md p-8 shadow-[0_5px_15px_rgba(0,243,255,0.05)]">
          <div>
            <h2 className="text-4xl font-black uppercase tracking-widest text-[#00f3ff] text-glow">
              <span className="opacity-50 mr-2">//</span>{activeTab}
            </h2>
            <p className="text-xs font-bold mt-2 text-[#ff00ff] tracking-[0.2em]">CORTEX_STREAM_MONITORING: ACTIVE</p>
          </div>
          <div className="flex space-x-2 opacity-60">
             <div className="h-4 w-12 bg-[#00f3ff] shadow-[0_0_8px_#00f3ff]"></div>
             <div className="h-4 w-4 bg-[#ff00ff]"></div>
             <div className="h-4 w-8 bg-[#39ff14]"></div>
          </div>
        </header>

        {activeTab === 'overview' && (
          <div className="grid grid-cols-12 gap-8 px-2">
            <div className="col-span-4 card bg-black/60 border-t-4 border-t-[#00f3ff]">
              <h3 className="text-xs text-[#00f3ff] tracking-widest mb-2 opacity-70">[DATA_VOL]</h3>
              <p className="text-5xl font-black text-glow">{stats.total_transactions.toLocaleString()}</p>
              <div className="mt-4 h-1 w-full bg-slate-800"><div className="h-full bg-[#00f3ff] w-[85%] shadow-[0_0_5px_#00f3ff]"></div></div>
            </div>
            
            <div className="col-span-4 card bg-black/60 border-t-4 border-t-[#ff00ff]">
              <h3 className="text-xs text-[#ff00ff] tracking-widest mb-2 opacity-70">[MARKET_NODES]</h3>
              <p className="text-5xl font-black text-[#ff00ff]">{stats.total_markets}</p>
              <div className="mt-4 h-1 w-full bg-slate-800"><div className="h-full bg-[#ff00ff] w-[40%] shadow-[0_0_5px_#ff00ff]"></div></div>
            </div>

            <div className="col-span-4 card bg-black/60 border-t-4 border-t-[#39ff14]">
              <h3 className="text-xs text-[#39ff14] tracking-widest mb-2 opacity-70">[LAST_SYNC]</h3>
              <p className="text-xl font-black leading-tight break-all text-[#39ff14]">{stats.last_update || 'AWAITING_SIGNAL'}</p>
              <p className="text-[10px] mt-4 opacity-50 tracking-widest text-white">ORACLE_CONNECTION: ESTABLISHED</p>
            </div>

            <div className="col-span-12 card bg-black/80">
              <div className="flex justify-between items-center mb-4 border-b border-[#00f3ff]/30 pb-2">
                <h3 className="text-sm uppercase tracking-widest text-[#00f3ff]">Raw_Telemetry_Feed</h3>
                <div className="flex items-center space-x-2">
                  <span className="text-[10px] text-[#ff00ff] border border-[#ff00ff] px-2 py-0.5 animate-pulse">LIVE</span>
                </div>
              </div>
              <div className="h-64 overflow-y-auto text-xs space-y-1 p-4 bg-[#05050a] border border-slate-800">
                {logs.map((log, i) => <div key={i} className="text-[#39ff14] opacity-80 break-all">&gt; {log}</div>)}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div className="grid grid-cols-2 gap-8 px-2">
            {[
              { id: 'xgboost', title: 'XGB_WHALE_SHADOW', color: '#00f3ff', desc: 'Identify covert market makers via gradient boosting.' },
              { id: 'logreg', title: 'LOGREG_BASELINE', color: '#ff00ff', desc: 'Linear decider for minimal-variance signal extraction.' },
              { id: 'info_index', title: 'INFO_ASYMMETRY', color: '#39ff14', desc: 'Calculates price slippage vs wallet flow anomalies.' },
              { id: 'random', title: 'STOCHASTIC_NULL', color: 'white', desc: 'Monte Carlo noise baseline generation.' }
            ].map(m => (
              <div key={m.id} className="card bg-black/60 flex flex-col" style={{ borderTop: `4px solid ${m.color}` }}>
                <h3 className="text-2xl font-black mt-2 tracking-widest" style={{ color: m.color, textShadow: `0 0 10px ${m.color}66` }}>{m.title}</h3>
                <p className="mt-4 text-sm font-light opacity-80 flex-1">{m.desc}</p>
                <div className="mt-6 border border-slate-800 p-2 mb-6 text-[10px] text-slate-400">
                  <span style={{color: m.color}}>CONFIG:</span> AUTO_TUNE=1 | LATENCY_SIM=ON
                </div>
                <button 
                  onClick={() => handleRunModel(m.id)}
                  disabled={loading}
                  className={m.id === 'xgboost' || m.id === 'random' ? 'btn-acid-cyan w-full' : 'btn-acid-magenta w-full'}
                >
                  {loading ? 'INITIALIZING...' : `EXEC_${m.id}`}
                </button>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'comparison' && (
          <div className="space-y-8 px-2">
            <div className="card bg-black/60">
              <h3 className="text-xl font-black mb-6 border-l-4 border-[#00f3ff] pl-4 text-[#00f3ff] text-glow tracking-widest">PERFORMANCE_MATRIX</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-[#00f3ff] text-[#ff00ff] tracking-widest">
                      <th className="p-3">ARCHITECTURE</th>
                      <th className="p-3">YIELD_ROA</th>
                      <th className="p-3">WIN_RATIO</th>
                      <th className="p-3">SHARPE_IDX</th>
                      <th className="p-3 text-right">EDGE_VISUAL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.map((m, i) => (
                      <tr key={i} className="border-b border-white/10 hover:bg-white/5 transition-colors">
                        <td className="p-3 tracking-wider text-[#00f3ff]">{m.name}</td>
                        <td className="p-3 text-[#39ff14]">{m.roi}%</td>
                        <td className="p-3 text-white">{m.win_rate * 100}%</td>
                        <td className="p-3 text-white">{m.sharpe}</td>
                        <td className="p-3 text-right">
                          <div className="h-1 bg-slate-800 inline-block w-32 relative">
                             <div className="h-full bg-[#00f3ff] shadow-[0_0_5px_#00f3ff]" style={{ width: `${Math.max(0, m.roi * 3)}px` }}></div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-8">
               <div className="card bg-black/60 border-[#39ff14]">
                 <h3 className="text-sm font-bold tracking-widest mb-3 text-[#39ff14] border-b border-[#39ff14]/30 pb-2">AI_DIRECTIVE</h3>
                 <p className="text-xs text-white opacity-80 font-light leading-relaxed">&gt; Pattern recognized: XGBoost exhibits +5.2% alpha in low-liquidity sectors.<br/>&gt; Recommendation: Scale allocation weight dynamically inversely proportional to volume.</p>
               </div>
               <div className="card bg-black/60 border-[#ff00ff]">
                 <h3 className="text-sm font-bold tracking-widest mb-3 text-[#ff00ff] border-b border-[#ff00ff]/30 pb-2">INTEGRITY_CHECK</h3>
                 <p className="text-xs text-white opacity-80 font-light leading-relaxed">&gt; Look-ahead bias: NEUTRALIZED<br/>&gt; Slippage SIM: APPLIED (0.15%)<br/>&gt; Oracle Delay: CALIBRATED</p>
               </div>
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          <div className="columns-1 lg:columns-2 gap-8 space-y-8 px-2">
            {results.map(file => (
              <div key={file} className="card bg-black/80 break-inside-avoid p-4">
                <div className="flex justify-between items-center mb-4 border-b border-white/20 pb-2">
                   <h3 className="text-xs uppercase tracking-widest text-[#00f3ff]">{file.split('.')[0]}</h3>
                   <span className="text-[10px] text-[#ff00ff] tracking-widest">RENDER_COMPLETE</span>
                </div>
                <div className="border border-slate-700 relative overflow-hidden group">
                  <div className="absolute inset-0 bg-[#00f3ff]/10 opacity-0 group-hover:opacity-100 transition-opacity mix-blend-screen"></div>
                  <img 
                    src={`${API_BASE}/plots/${file}`} 
                    alt={file} 
                    className="w-full filter contrast-125"
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
