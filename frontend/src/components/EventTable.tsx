import React from 'react';
import { CalendarDays, ExternalLink, Activity } from 'lucide-react';

interface EventInfo {
    evento: string;
    categoria: string;
    proveedor?: string;
    fecha: string;
    hora_argentina: string;
    descripcion: string;
    impacto_estimado: string;
    fuente: string;
}

interface EventTableProps {
    events: EventInfo[];
}

const EventTable: React.FC<EventTableProps> = ({ events }) => {
    if (!events || events.length === 0) {
        return (
            <div className="w-full p-8 text-center bg-slate-800/50 rounded-xl border border-slate-700/50 text-slate-400">
                No se encontraron eventos relevantes para esta fecha.
            </div>
        );
    }

    const getImpactColor = (impacto: string) => {
        const imp = impacto?.toLowerCase() || '';
        if (imp.includes('alto')) return 'bg-rose-500/20 text-rose-300 border-rose-500/30';
        if (imp.includes('medio')) return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
    };

    const getCategoryColor = (cat: string) => {
        const c = cat?.toLowerCase() || '';
        if (c.includes('deporte')) return 'text-blue-400';
        if (c.includes('gaming')) return 'text-purple-400';
        if (c.includes('streaming')) return 'text-pink-400';
        return 'text-cyan-400';
    };

    return (
        <div className="w-full bg-slate-900 border border-slate-700 rounded-xl overflow-hidden shadow-2xl">
            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="bg-slate-800/80 text-slate-300 text-sm uppercase tracking-wider">
                            <th className="p-4 font-semibold">Evento</th>
                            <th className="p-4 font-semibold">Proveedor</th>
                            <th className="p-4 font-semibold">Horario (ART)</th>
                            <th className="p-4 font-semibold">Impacto</th>
                            <th className="p-4 font-semibold">Fuente</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                        {events.map((ev, idx) => (
                            <tr key={idx} className="hover:bg-slate-800/30 transition-colors group">
                                <td className="p-4">
                                    <div className="font-medium text-white group-hover:text-emerald-400 transition-colors">
                                        {ev.evento}
                                    </div>
                                    <div className="text-sm text-slate-400 mt-1 flex items-center gap-2">
                                        <span className={`font-semibold ${getCategoryColor(ev.categoria)}`}>
                                            {ev.categoria}
                                        </span>
                                        <span className="opacity-50">•</span>
                                        <span className="truncate max-w-xs block" title={ev.descripcion}>
                                            {ev.descripcion}
                                        </span>
                                    </div>
                                </td>
                                <td className="p-4 whitespace-nowrap">
                                    <span className="text-slate-200 font-medium px-2 py-1 bg-slate-800 rounded-md border border-slate-700 shadow-sm">
                                        {ev.proveedor || 'N/A'}
                                    </span>
                                </td>
                                <td className="p-4 whitespace-nowrap">
                                    <div className="flex items-center gap-2 text-slate-300">
                                        <CalendarDays size={16} className="text-slate-500" />
                                        <span>{ev.hora_argentina}</span>
                                    </div>
                                </td>
                                <td className="p-4 whitespace-nowrap">
                                    <span className={`px-3 py-1 pb-1.5 rounded-full text-xs font-bold border flex items-center gap-1.5 w-max ${getImpactColor(ev.impacto_estimado)}`}>
                                        <Activity size={14} />
                                        {ev.impacto_estimado}
                                    </span>
                                </td>
                                <td className="p-4">
                                    {ev.fuente && ev.fuente !== 'N/A' ? (
                                        <a
                                            href={ev.fuente}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-emerald-500 hover:text-emerald-400 p-2 hover:bg-emerald-500/10 rounded-lg inline-flex transition-colors"
                                            title="Abrir fuente"
                                        >
                                            <ExternalLink size={18} />
                                        </a>
                                    ) : (
                                        <span className="text-slate-600 text-sm">N/A</span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default EventTable;
