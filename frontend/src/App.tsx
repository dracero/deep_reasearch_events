import React, { useState } from 'react';
import { Search } from 'lucide-react';
import LoadingState from './components/LoadingState';
import SearchPlan from './components/SearchPlan';
import EventTable from './components/EventTable';
import ErrorState from './components/ErrorState';

// Interfaz para definir el mensaje A2UI
interface A2UIEvent {
  type: string;
  component: string;
  props: any;
}

function App() {
  const [dateStr, setDateStr] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  // Guardamos un historial de los componentes que el agente va devolviendo en formato A2UI
  const [agentComponents, setAgentComponents] = useState<A2UIEvent[]>([]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!dateStr) return;

    setIsSearching(true);
    // Limpiar resultados anteriores
    setAgentComponents([]);

    try {
      const response = await fetch('http://localhost:8000/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date: dateStr })
      });

      if (!response.body) throw new Error("No hay stream en la respuesta");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE lines
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; // Keep the incomplete line in the buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            try {
              const event: A2UIEvent = JSON.parse(dataStr);
              // Como este es A2UI, agregamos la interfaz que el agente solicita al flujo (chat/timeline)
              setAgentComponents(prev => [...prev, event]);
            } catch (err) {
              console.error("Error parseando evento A2UI:", err);
            }
          }
        }
      }
    } catch (error: any) {
      setAgentComponents(prev => [...prev, { type: 'ui', component: 'ErrorState', props: { error: error.message } }]);
    } finally {
      setIsSearching(false);
    }
  };

  // Renderizador dinámico del protocolo A2UI
  const renderA2UIComponent = (event: A2UIEvent, index: number) => {
    switch (event.component) {
      case 'LoadingState':
        return <LoadingState key={index} {...event.props} />;
      case 'SearchPlan':
        return <SearchPlan key={index} {...event.props} />;
      case 'EventTable':
        return <EventTable key={index} {...event.props} />;
      case 'ErrorState':
        return <ErrorState key={index} {...event.props} />;
      default:
        return (
          <div key={index} className="p-4 bg-slate-800 rounded-lg text-slate-400 text-sm my-2">
            Componente Desconocido: {event.component}
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6">
      <div className="max-w-4xl mx-auto space-y-8">

        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent pb-2">
            Deep Research Argentina
          </h1>
          <p className="text-slate-400 max-w-xl mx-auto text-lg">
            Investigador autónomo de eventos generadores de tráfico de internet basados en A2UI.
          </p>
        </div>

        {/* Formulario Principal */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6 shadow-xl relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-cyan-500/5 pointer-events-none" />
          <form onSubmit={handleSearch} className="relative flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <input
                type="date"
                required
                value={dateStr}
                onChange={(e) => setDateStr(e.target.value)}
                className="w-full bg-slate-900/80 border border-slate-600 rounded-xl px-4 py-4 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
                disabled={isSearching}
              />
            </div>
            <button
              type="submit"
              disabled={isSearching}
              className="px-8 py-4 bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-slate-900 font-bold rounded-xl transition-all flex items-center justify-center gap-2 uppercase tracking-wide"
            >
              {isSearching ? (
                <div className="h-5 w-5 border-2 border-slate-400 border-t-slate-900 rounded-full animate-spin" />
              ) : (
                <>
                  <Search size={20} />
                  Investigar
                </>
              )}
            </button>
          </form>
        </div>

        {/* Zona de Renderizado A2UI */}
        <div className="space-y-6 flex flex-col items-center">
          {agentComponents.map((componentEvent, index) => renderA2UIComponent(componentEvent, index))}
        </div>

      </div>
    </div>
  );
}

export default App;
