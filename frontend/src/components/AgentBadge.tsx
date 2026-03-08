import React from 'react';
import { Bot } from 'lucide-react';

interface Props {
    agent_name: string;
}

const AgentBadge: React.FC<Props> = ({ agent_name }) => {
    // Configuración de colores dinámica
    const isExplainer = agent_name.toLowerCase().includes('explainer');
    const gradient = isExplainer ? "from-blue-500 to-indigo-600" : "from-emerald-500 to-teal-600";
    const bgBadge = isExplainer ? "bg-blue-500/10 text-blue-400 border-blue-500/30" : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";

    return (
        <div className={`w-full max-w-4xl flex items-center gap-3 p-4 mb-2 rounded-xl border bg-slate-800/60 shadow-lg ${bgBadge}`}>
            <div className={`p-2 rounded-lg bg-gradient-to-br ${gradient} text-white shadow-md`}>
                <Bot size={24} />
            </div>
            <div className="flex flex-col">
                <span className="text-xs font-bold tracking-wider uppercase opacity-75">Respondiendo</span>
                <span className="text-lg font-semibold">{agent_name}</span>
            </div>
        </div>
    );
};

export default AgentBadge;
