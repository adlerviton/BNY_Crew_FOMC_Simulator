// frontend/components/SimulationProgress.tsx
"use client"

import React from 'react';
import { taskNames } from '../lib/types';

interface SimulationProgressProps {
    status: 'idle' | 'running' | 'completed' | 'error';
    currentTask: string | null;
    tasksCompleted: string[];
    progress: number;
    error: string | null;
}

const SimulationProgress: React.FC<SimulationProgressProps> = ({
    status,
    currentTask,
    tasksCompleted,
    progress,
    error
}) => {
    // Get friendly task name
    const getFriendlyTaskName = (taskKey: string) => {
        return taskNames[taskKey] || taskKey;
    };

    return (
        <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium">Simulation Progress</h3>
                <div className="flex items-center gap-2">
                    <div
                        className={`w-3 h-3 rounded-full ${status === 'idle' ? 'bg-gray-400' :
                                status === 'running' ? 'bg-blue-500 animate-pulse' :
                                    status === 'completed' ? 'bg-green-500' :
                                        'bg-red-500'
                            }`}
                    />
                    <span className="text-sm capitalize">{status}</span>
                </div>
            </div>

            {/* Progress bar */}
            <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div
                    className={`h-2.5 rounded-full ${status === 'error' ? 'bg-red-500' : 'bg-blue-600'}`}
                    style={{ width: `${progress}%` }}
                ></div>
            </div>

            {/* Current task */}
            {currentTask && status === 'running' && (
                <div className="flex items-center gap-2">
                    <svg className="animate-spin h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>In progress: {getFriendlyTaskName(currentTask)}</span>
                </div>
            )}

            {/* Error message */}
            {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                    <p className="font-bold">Error</p>
                    <p>{error}</p>
                </div>
            )}

            {/* Tasks list */}
            <div className="flex flex-col gap-1 max-h-60 overflow-y-auto mt-2">
                {tasksCompleted.map((task, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm">
                        <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        <span>{getFriendlyTaskName(task)}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default SimulationProgress;