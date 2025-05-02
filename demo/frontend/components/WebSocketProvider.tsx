// frontend/components/WebSocketProvider.tsx
"use client"

import React, { createContext, useContext, ReactNode } from 'react';
import { useSimulationData } from '../lib/useSimulationData';
import { SimulationState, initialSimulationState } from '../lib/types';

interface SimulationContextType {
    simulationData: SimulationState;
    isConnected: boolean;
    startSimulation: () => void;
    resetSimulation: () => void;
}

const SimulationContext = createContext<SimulationContextType>({
    simulationData: initialSimulationState,
    isConnected: false,
    startSimulation: () => { },
    resetSimulation: () => { },
});

// Custom hook to use the simulation context
export const useSimulation = () => useContext(SimulationContext);

// Provider component
export const WebSocketProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const {
        simulationData,
        isConnected,
        startSimulation,
        resetSimulation,
    } = useSimulationData();

    return (
        <SimulationContext.Provider
            value={{
                simulationData,
                isConnected,
                startSimulation,
                resetSimulation,
            }}
        >
            {children}
        </SimulationContext.Provider>
    );
};