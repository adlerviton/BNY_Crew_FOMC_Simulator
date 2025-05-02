// frontend/lib/useSimulationData.ts
import { useState, useEffect, useCallback, useRef } from 'react';
import { SimulationState, initialSimulationState } from './types';

export function useSimulationData() {
    const [simulationData, setSimulationData] = useState<SimulationState>(initialSimulationState);
    const [isConnected, setIsConnected] = useState(false);
    const socketRef = useRef<WebSocket | null>(null);

    // Set up WebSocket connection
    useEffect(() => {
        // Create WebSocket connection
        const socket = new WebSocket('ws://localhost:8000/ws');
        socketRef.current = socket;

        // Connection opened
        socket.addEventListener('open', () => {
            console.log('WebSocket connected');
            setIsConnected(true);
        });

        // Listen for messages
        socket.addEventListener('message', (event) => {
            try {
                const data = JSON.parse(event.data);
                setSimulationData(data);
            } catch (error) {
                console.error('Error parsing WebSocket data:', error);
            }
        });

        // Connection closed
        socket.addEventListener('close', () => {
            console.log('WebSocket disconnected');
            setIsConnected(false);
        });

        // Connection error
        socket.addEventListener('error', (error) => {
            console.error('WebSocket error:', error);
            setIsConnected(false);
        });

        // Clean up on unmount
        return () => {
            socket.close();
        };
    }, []);

    // Function to start the simulation
    const startSimulation = useCallback(() => {
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({ action: 'start_simulation' }));
        } else {
            console.error('WebSocket not connected');
        }
    }, []);

    // Function to reset the simulation
    const resetSimulation = useCallback(() => {
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({ action: 'reset_simulation' }));
        } else {
            console.error('WebSocket not connected');
        }
    }, []);

    // Fallback to REST API if WebSocket is not available
    const fetchDataFromAPI = useCallback(async () => {
        try {
            const response = await fetch('http://localhost:8000/api/status');
            const data = await response.json();
            setSimulationData(data);
        } catch (error) {
            console.error('Error fetching data from API:', error);
        }
    }, []);

    // Start simulation via REST API
    const startSimulationViaAPI = useCallback(async () => {
        try {
            await fetch('http://localhost:8000/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            // Fetch initial state after starting
            fetchDataFromAPI();
        } catch (error) {
            console.error('Error starting simulation via API:', error);
        }
    }, [fetchDataFromAPI]);

    // Reset simulation via REST API
    const resetSimulationViaAPI = useCallback(async () => {
        try {
            await fetch('http://localhost:8000/api/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            // Fetch initial state after resetting
            fetchDataFromAPI();
        } catch (error) {
            console.error('Error resetting simulation via API:', error);
        }
    }, [fetchDataFromAPI]);

    // Combined functions that try WebSocket first, then fall back to REST API
    const startSimulationWithFallback = useCallback(() => {
        if (isConnected) {
            startSimulation();
        } else {
            startSimulationViaAPI();
        }
    }, [isConnected, startSimulation, startSimulationViaAPI]);

    const resetSimulationWithFallback = useCallback(() => {
        if (isConnected) {
            resetSimulation();
        } else {
            resetSimulationViaAPI();
        }
    }, [isConnected, resetSimulation, resetSimulationViaAPI]);

    return {
        simulationData,
        isConnected,
        startSimulation: startSimulationWithFallback,
        resetSimulation: resetSimulationWithFallback,
    };
}