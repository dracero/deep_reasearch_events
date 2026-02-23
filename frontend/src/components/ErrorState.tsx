import React from 'react';
import { AlertCircle } from 'lucide-react';

interface ErrorStateProps {
    error: string;
}

const ErrorState: React.FC<ErrorStateProps> = ({ error }) => {
    return (
        <div className="w-full bg-red-500/10 border border-red-500/20 text-red-200 p-4 rounded-xl flex items-start gap-3 backdrop-blur-sm">
            <AlertCircle className="shrink-0 mt-0.5" size={20} />
            <div>
                <h3 className="font-semibold text-red-300">Algo salió mal</h3>
                <p className="opacity-80 text-sm mt-1">{error}</p>
            </div>
        </div>
    );
};

export default ErrorState;
