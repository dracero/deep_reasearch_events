import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
    message: string;
}

const LoadingState: React.FC<LoadingStateProps> = ({ message }) => {
    return (
        <div className="w-full flex items-center justify-center p-6 bg-slate-800/50 border border-slate-700/50 rounded-xl gap-3 text-emerald-400">
            <Loader2 className="animate-spin" size={24} />
            <span className="font-medium animate-pulse">{message}</span>
        </div>
    );
};

export default LoadingState;
