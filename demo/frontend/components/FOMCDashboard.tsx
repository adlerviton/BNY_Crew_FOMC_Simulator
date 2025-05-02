// frontend/components/FOMCDashboard.tsx
"use client"

import React, { useState, useEffect } from 'react';
import { useSimulation } from './WebSocketProvider';
import VoteChart from './VoteChart';
import PredictionChart from './PredictionChart';
import SimulationProgress from './SimulationProgress';
import FOMCStatement from './FOMCStatement';
import ConversationLog from './ConversationLog';

const FOMCDashboard: React.FC = () => {
    const { simulationData, isConnected, startSimulation, resetSimulation } = useSimulation();
    const [activeTab, setActiveTab] = useState<'dashboard' | 'conversations'>('dashboard');
    const [availableDates, setAvailableDates] = useState<string[]>([]);
    const [selectedDate, setSelectedDate] = useState<string>('');

    useEffect(() => {
        // Fetch available dates when component mounts
        fetch('http://localhost:8000/api/available-dates')
            .then(response => response.json())
            .then(data => {
                // Sort dates from most recent to oldest
                const sortedDates = data.dates.sort((a: string, b: string) => {
                    const [yearA, monthA] = a.split('_').map(Number);
                    const [yearB, monthB] = b.split('_').map(Number);
                    if (yearA !== yearB) return yearB - yearA;
                    return monthB - monthA;
                });
                setAvailableDates(sortedDates);
                setSelectedDate('');
            })
            .catch(error => console.error('Error fetching dates:', error));
    }, []);

    // Function to format date for display (e.g., "23_3" -> "March 2023")
    const formatDate = (dateStr: string) => {
        const [year, month] = dateStr.split('_');
        const months = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ];
        return `${months[parseInt(month) - 1]} 20${year}`;
    };

    const handleDateChange = async (date: string) => {
        setSelectedDate(date);
        try {
            const response = await fetch('http://localhost:8000/api/set-date', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ date }),
            });

            if (!response.ok) {
                const error = await response.json();
                console.error('Error setting date:', error.message);
            }
        } catch (error) {
            console.error('Error setting date:', error);
        }
    };

    return (
        <div className="flex flex-col gap-6">
            {/* Connection status, date selection, and actions */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-gray-50 p-4 rounded-lg shadow">
                <div className="flex flex-col gap-4 w-full md:w-auto">
                    <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                        <span>{isConnected ? 'Connected to server' : 'Disconnected from server'}</span>
                    </div>
                    <div className="flex items-center gap-2 w-full md:w-64">
                        <select
                            value={selectedDate}
                            onChange={(e) => handleDateChange(e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        >
                            <option value="">Choose date</option>
                            {availableDates.map((date) => (
                                <option key={date} value={date}>
                                    {formatDate(date)}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                <div className="flex gap-3">
                    <button
                        onClick={startSimulation}
                        disabled={simulationData.status === 'running' || !selectedDate}
                        className={`px-4 py-2 rounded ${simulationData.status === 'running' || !selectedDate
                            ? 'bg-gray-300 cursor-not-allowed'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                            }`}
                    >
                        {simulationData.status === 'running'
                            ? 'Simulation Running...'
                            : !selectedDate
                                ? 'Select a date first'
                                : 'Start Simulation'}
                    </button>

                    <button
                        onClick={resetSimulation}
                        disabled={simulationData.status === 'running'}
                        className={`px-4 py-2 border rounded ${simulationData.status === 'running'
                            ? 'border-gray-300 text-gray-300 cursor-not-allowed'
                            : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                            }`}
                    >
                        Reset
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-gray-200">
                <button
                    className={`py-3 px-6 font-medium ${activeTab === 'dashboard'
                        ? 'border-b-2 border-blue-500 text-blue-600'
                        : 'text-gray-500 hover:text-gray-700'
                        }`}
                    onClick={() => setActiveTab('dashboard')}
                >
                    Dashboard
                </button>
                <button
                    className={`py-3 px-6 font-medium ${activeTab === 'conversations'
                        ? 'border-b-2 border-blue-500 text-blue-600'
                        : 'text-gray-500 hover:text-gray-700'
                        }`}
                    onClick={() => setActiveTab('conversations')}
                >
                    Conversations
                </button>
            </div>

            {/* Dashboard Content */}
            {activeTab === 'dashboard' && (
                <>
                    {/* Progress tracking */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="col-span-1 bg-white p-4 rounded-lg shadow">
                            <SimulationProgress
                                status={simulationData.status}
                                currentTask={simulationData.currentTask}
                                tasksCompleted={simulationData.tasksCompleted}
                                progress={simulationData.progress}
                                error={simulationData.error}
                            />
                        </div>

                        {/* Visualizations */}
                        <div className="col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="bg-white p-4 rounded-lg shadow h-80">
                                <h3 className="text-lg font-medium mb-4">Interest Rate Votes</h3>
                                <VoteChart data={simulationData.votes} />
                            </div>

                            <div className="bg-white p-4 rounded-lg shadow h-80">
                                <h3 className="text-lg font-medium mb-4">2025 Rate Predictions</h3>
                                <PredictionChart data={simulationData.predictions} />
                            </div>
                        </div>
                    </div>

                    {/* FOMC statement */}
                    <div className="mt-4">
                        <FOMCStatement statement={simulationData.fomcStatement} />
                    </div>
                </>
            )}

            {/* Conversations Content */}
            {activeTab === 'conversations' && (
                <div className="bg-white rounded-lg shadow-lg overflow-hidden">
                    <ConversationLog logs={simulationData.conversationLogs} />
                </div>
            )}
        </div>
    );
};

export default FOMCDashboard;