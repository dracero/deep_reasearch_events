import React, { useState } from 'react';
import { Send, Bot } from 'lucide-react';
import LoadingState from './components/LoadingState';
import SearchPlan from './components/SearchPlan';
import EventTable from './components/EventTable';
import ErrorState from './components/ErrorState';
import TravelRoutes from './components/TravelRoutes';
import AgentBadge from './components/AgentBadge';
import AgentQuestion from './components/AgentQuestion';

// Interfaz para definir el mensaje A2UI
interface A2UIEvent {
  type: string;
  component: string;
  props: any;
}

function App() {
  const [chatMessage, setChatMessage] = useState('');
  const [isSearching, setIsSearching] = useState(false);

  // Conversation context tracking — used to bypass the intent router on follow-ups
  const [activeAgent, setActiveAgent] = useState<string>('');       // 'viajes' | 'eventos' | ''
  const [travelContext, setTravelContext] = useState<Record<string, string>>({});

  // Guardamos el historial de la conversación {role: 'user' | 'assistant', content: string}
  const [chatHistory, setChatHistory] = useState<{ role: string, content: string }[]>([]);

  // Guardamos un historial de los componentes que el agente va devolviendo en formato A2UI
  const [agentComponents, setAgentComponents] = useState<A2UIEvent[]>([]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatMessage) return;

    setIsSearching(true);
    // Añadimos el mensaje del usuario a la vista
    setAgentComponents(prev => [
      ...prev,
      { type: 'ui', component: 'UserChat', props: { message: chatMessage } }
    ]);
    const sendMsg = chatMessage;

    // Agregamos al historial
    const userMsgObj = { role: 'user', content: sendMsg };
    const currentHistory = [...chatHistory];
    setChatHistory(prev => [...prev, userMsgObj]);

    setChatMessage('');

    // Determine if this is a follow-up (active agent set) or a new search
    const isFollowUp = activeAgent !== '';
    // If it's a brand-new query (no active agent), reset context
    if (!isFollowUp) {
      setTravelContext({});
    }

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: sendMsg,
          history: currentHistory, // Enviamos el historial previo
          active_agent: activeAgent,  // Tells the orchestrator to bypass LLM router
          travel_context: travelContext,
        }),
        // No AbortController/signal — let the stream run as long as the backend needs
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
          // Skip SSE heartbeat/comment lines (e.g. ": keep-alive")
          if (line.startsWith(':') || !line.trim()) continue;

          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6).trim();
            if (!dataStr) continue;
            try {
              const event: A2UIEvent = JSON.parse(dataStr);
              setAgentComponents(prev => [...prev, event]);

              // Track which agent is now active so follow-ups are routed correctly
              if (event.component === 'AgentBadge') {
                const agentName: string = (event.props.agent_name || '').toLowerCase();
                if (agentName.includes('viajes')) {
                  setActiveAgent('viajes');
                } else if (agentName.includes('eventos')) {
                  setActiveAgent('eventos');
                }
              }
              // When an AgentQuestion appears, the agent is waiting for a follow-up
              if (event.component === 'AgentQuestion') {
                setActiveAgent('viajes'); // Only the travel agent asks questions currently
              }
              // Accumulate travel context from orchestrator ContextUpdate events
              if (event.component === 'ContextUpdate' && event.props?.context) {
                setTravelContext(prev => ({ ...prev, ...event.props.context }));
              }
              // When results arrive, the conversation is complete — clear active agent
              if (event.component === 'TravelRoutes' || event.component === 'EventTable') {
                setActiveAgent('');
              }
            } catch (err) {
              console.error("Error parseando evento A2UI:", err, "raw:", dataStr);
            }
          }
        }
      }

      // Al terminar el stream, agregamos una nota estructural al historial
      // para que el LLM sepa que el agente le dio una respuesta al usuario.
      setChatHistory(prev => [
        ...prev,
        { role: 'assistant', content: '[Respuesta del agente renderizada en la Interfaz de Usuario]' }
      ]);

    } catch (error: any) {
      const isRateLimit = error?.message?.toLowerCase().includes('rate') || error?.message?.includes('429');
      const displayMsg = isRateLimit
        ? 'La API de Groq alcanzó el límite de requests. El sistema reintentará automáticamente — por favor esperá unos segundos e intentá de nuevo.'
        : error.message;
      setAgentComponents(prev => [...prev, { type: 'ui', component: 'ErrorState', props: { error: displayMsg } }]);
    } finally {
      setIsSearching(false);
    }
  };

  // Renderizador dinámico del protocolo A2UI
  const renderA2UIComponent = (event: A2UIEvent, index: number) => {
    switch (event.component) {
      case 'UserChat':
        return (
          <div key={index} className="w-full max-w-[1200px] flex justify-end my-4 mx-auto">
            <div className="p-4 bg-emerald-600/30 text-emerald-100 border border-emerald-500/50 rounded-2xl rounded-tr-none shadow-lg max-w-[80%]">
              {event.props.message}
            </div>
          </div>
        );
      case 'AgentBadge':
        return <AgentBadge key={index} {...event.props} />;
      case 'LoadingState':
        return <LoadingState key={index} {...event.props} />;
      case 'SearchPlan':
        return <SearchPlan key={index} {...event.props} />;
      case 'EventTable':
        return <EventTable key={index} {...event.props} />;
      case 'TravelRoutes':
        return <TravelRoutes key={index} {...event.props} />;
      case 'AgentQuestion':
        return <AgentQuestion key={index} {...event.props} />;
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
    <div className="min-h-screen bg-slate-900 text-white flex flex-col items-center">
      {/* Header Fijo */}
      <div className="w-full w-[95%] max-w-[1600px] py-6 px-4 bg-slate-900/90 backdrop-blur top-0 sticky z-10 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Bot size={32} className="text-emerald-500" />
          <h1 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent">
            BeeAI Orchestrator
          </h1>
        </div>
        <p className="text-slate-400 text-sm hidden md:block">
          Conectado vía Google A2A Protocol
        </p>
      </div>

      {/* Zona de Renderizado Chat / A2UI */}
      <div className="flex-1 w-[95%] max-w-[1600px] overflow-y-auto p-4 space-y-6 flex flex-col pb-32">
        {agentComponents.length === 0 ? (
          <div className="text-center mt-32 space-y-4 flex-col flex items-center">
            <Bot size={80} className="text-slate-700" />
            <p className="text-slate-400 text-xl font-medium">¿A dónde vamos o qué investigamos hoy?</p>
            <p className="text-slate-500 max-w-xl text-center">
              Ej: "Buscando vuelos baratos a Europa en septiembre"<br />o<br /> "Eventos de esports en Buenos Aires el 10/10/2026"
            </p>
          </div>
        ) : (
          agentComponents.map((componentEvent, index) => renderA2UIComponent(componentEvent, index))
        )}
      </div>

      {/* Formulario Inferior Fijo */}
      <div className="w-full bg-slate-800/80 backdrop-blur-md border-t border-slate-700/50 p-4 fixed bottom-0 left-0 right-0 z-20 shadow-[0_-10px_30px_rgba(0,0,0,0.5)]">
        <div className="w-[95%] max-w-[1600px] mx-auto">
          <form onSubmit={handleSearch} className="relative flex gap-2 w-full">
            <input
              type="text"
              required
              placeholder="Escribí tu consulta para los agentes..."
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              className="flex-1 bg-slate-900/80 border border-slate-600 rounded-xl px-4 py-4 pr-16 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all"
              disabled={isSearching}
            />
            <button
              type="submit"
              disabled={isSearching}
              className="absolute right-2 top-2 bottom-2 aspect-square flex items-center justify-center bg-emerald-500 hover:bg-emerald-400 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-slate-900 font-bold rounded-lg transition-all"
            >
              {isSearching ? (
                <div className="h-5 w-5 border-2 border-slate-400 border-t-slate-900 rounded-full animate-spin" />
              ) : (
                <Send size={20} />
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
