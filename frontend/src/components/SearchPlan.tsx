import React from 'react';
import { ListChecks } from 'lucide-react';

interface SearchPlanProps {
    queries: string[];
}

const SearchPlan: React.FC<SearchPlanProps> = ({ queries }) => {
    return (
        <div className="w-full bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 shadow-lg backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-4 text-emerald-400">
                <ListChecks size={24} />
                <h2 className="text-xl font-bold text-white">Plan de Búsqueda Generado</h2>
            </div>
            <ul className="space-y-2">
                {queries.map((q, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-slate-300">
                        <span className="text-emerald-500 font-mono mt-0.5">•</span>
                        {q}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default SearchPlan;
