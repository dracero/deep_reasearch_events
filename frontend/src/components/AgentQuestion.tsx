import React from "react";

interface AgentQuestionProps {
    question: string;
}

/**
 * Renders a clarifying question from the travel agent, styled like a
 * chat bubble from the agent side. Appears before the search results.
 */
const AgentQuestion: React.FC<AgentQuestionProps> = ({ question }) => {
    return (
        <div className="flex items-start gap-3 my-4 animate-fade-in">
            {/* Agent avatar */}
            <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg">
                <span className="text-white text-sm font-bold">✈</span>
            </div>

            {/* Bubble */}
            <div className="relative max-w-xl bg-slate-800 border border-blue-500/30 text-slate-100 rounded-2xl rounded-tl-none px-5 py-3 shadow-lg shadow-blue-900/20">
                {/* Label */}
                <p className="text-xs text-blue-400 font-semibold mb-1 uppercase tracking-widest">
                    Agente de Viajes
                </p>
                <p className="text-sm leading-relaxed">{question}</p>

                {/* Decorative corner */}
                <span className="absolute -left-2 top-0 w-4 h-4 overflow-hidden block">
                    <span className="absolute top-0 left-0 w-4 h-4 bg-slate-800 border-l border-t border-blue-500/30 rounded-bl-full" />
                </span>
            </div>
        </div>
    );
};

export default AgentQuestion;
