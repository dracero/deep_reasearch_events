import React from 'react';

interface RouteOption {
    ranking: number;
    ruta: string;
    origen_real?: string;
    destino_real?: string;
    horario_salida?: string;
    horario_llegada?: string;
    transporte: string;
    precio_usd: string;
    duracion_total: string;
    escalas: string;
    tipo: string;
    notas: string;
    fuente: string;
}

interface Props {
    routes: RouteOption[];
}

const TravelRoutes: React.FC<Props> = ({ routes }) => {
    if (!routes || routes.length === 0) {
        return (
            <div className="p-4 bg-slate-800 rounded-lg text-slate-400">
                No se encontraron rutas para estos parámetros.
            </div>
        );
    }

    return (
        <div className="w-full bg-slate-800/80 rounded-2xl overflow-hidden border border-slate-700/50 shadow-2xl">
            <div className="p-4 bg-slate-800 border-b border-slate-700/50">
                <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                    Rutas de Viaje Recomendadas
                </h3>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-300">
                    <thead className="text-xs uppercase bg-slate-900/50 text-slate-400">
                        <tr>
                            <th className="px-4 py-3">#</th>
                            <th className="px-4 py-3">Ruta</th>
                            <th className="px-4 py-3">Origen / Destino</th>
                            <th className="px-4 py-3">Horarios</th>
                            <th className="px-4 py-3">Transporte</th>
                            <th className="px-4 py-3 text-emerald-400 text-right">Precio</th>
                            <th className="px-4 py-3">Duración</th>
                            <th className="px-4 py-3">Escalas</th>
                            <th className="px-4 py-3">Detalles</th>
                        </tr>
                    </thead>
                    <tbody>
                        {routes.map((route, idx) => (
                            <tr key={idx} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                                <td className="px-4 py-3 font-bold text-slate-500">{route.ranking || idx + 1}</td>
                                <td className="px-4 py-3 font-medium text-slate-200">{route.ruta}</td>
                                <td className="px-4 py-3">
                                    <div className="flex flex-col">
                                        <span className="text-slate-200 font-medium whitespace-nowrap">{route.origen_real || '-'}</span>
                                        <span className="text-slate-500 whitespace-nowrap">→ {route.destino_real || '-'}</span>
                                    </div>
                                </td>
                                <td className="px-4 py-3">
                                    <div className="flex flex-col whitespace-nowrap">
                                        <span className="text-indigo-300">{route.horario_salida || '-'}</span>
                                        <span className="text-indigo-400/80">{route.horario_llegada || '-'}</span>
                                    </div>
                                </td>
                                <td className="px-4 py-3">{route.transporte}</td>
                                <td className="px-4 py-3 font-bold text-emerald-400 text-right">{route.precio_usd}</td>
                                <td className="px-4 py-3 text-slate-400">{route.duracion_total}</td>
                                <td className="px-4 py-3">
                                    <span className={`px-2 py-1 rounded text-xs ${route.escalas.toLowerCase().includes('directo') ? 'bg-indigo-500/20 text-indigo-300' : 'bg-slate-700 text-slate-300'
                                        }`}>
                                        {route.escalas.length > 30 ? route.escalas.substring(0, 30) + '...' : route.escalas}
                                    </span>
                                </td>
                                <td className="px-4 py-3">
                                    <div className="flex flex-col gap-1">
                                        <span className="text-xs text-blue-400">{route.tipo}</span>
                                        {route.fuente && (
                                            <a href={route.fuente} target="_blank" rel="noopener noreferrer" className="text-xs text-slate-500 hover:text-blue-400 underline decoration-slate-600 underline-offset-2">
                                                Ver fuente
                                            </a>
                                        )}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default TravelRoutes;
