// frontend/components/ConversationLog.tsx
"use client"

import React, { useRef, useEffect } from 'react';

interface ConversationEntry {
    type: string;
    agent?: string;
    task?: string;
    content?: string;
    timestamp: number;
    description?: string;
    vote?: string;
    prediction?: string;
}

interface ConversationLogProps {
    logs: ConversationEntry[];
}

const ConversationLog: React.FC<ConversationLogProps> = ({ logs }) => {
    const scrollRef = useRef<HTMLDivElement>(null);

    // Scroll to bottom when logs update
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    // Format timestamp
    const formatTime = (timestamp: number) => {
        const date = new Date(timestamp * 1000);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    };

    // Get agent color based on name
    const getAgentColor = (agent?: string) => {
        if (!agent) return "text-gray-600";

        switch (agent) {
            case "Regional Pragmatist":
            case "Regional_Pragmatists":
                return "text-blue-600";
            case "Academic Balancer":
            case "Academic_Balancers":
                return "text-green-600";
            case "Central Policymaker":
            case "Central_Policymakers":
                return "text-red-600";
            case "economist":
                return "text-yellow-600";
            case "analyst":
                return "text-purple-600";
            default:
                return "text-gray-600";
        }
    };

    // Get friendly task name
    const getFriendlyTaskName = (taskKey: string) => {
        const taskNames: Record<string, string> = {
            "probabilities_comment": "Analyzing Market Probabilities",
            "get_economic_suggestions": "Generating Economic Policy Suggestions",
            "regional_analysis": "Regional Pragmatists Analysis",
            "academic_analysis": "Academic Balancers Analysis",
            "central_analysis": "Central Policymakers Analysis",
            "regional_discussion": "Regional Pragmatists Discussion",
            "academic_discussion": "Academic Balancers Discussion",
            "central_discussion": "Central Policymakers Discussion",
            "regional_vote": "Regional Pragmatists Voting",
            "academic_vote": "Academic Balancers Voting",
            "central_vote": "Central Policymakers Voting",
            "other_summary": "Preparing FOMC Statement",
            "vote_summary": "Summarizing Votes",
            "prediction_summary": "Summarizing Predictions",
            "summary_final": "Finalizing Results"
        };

        return taskNames[taskKey] || taskKey;
    };

    // Render log entry based on type
    const renderLogEntry = (entry: ConversationEntry, index: number) => {
        const timeString = formatTime(entry.timestamp);
        const agentColor = getAgentColor(entry.agent);

        switch (entry.type) {
            case "task_start":
                return (
                    <div key={index} className="p-3 border-b border-gray-200">
                        <div className="flex items-start">
                            <div className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium mr-2">
                                {timeString}
                            </div>
                            <div>
                                <span className="font-semibold">New Task Started:</span> {entry.task ? getFriendlyTaskName(entry.task) : "Unknown Task"}
                                <div className="mt-1 text-sm text-gray-600">
                                    <span className={`font-medium ${agentColor}`}>
                                        {entry.agent?.replace("_", " ")}
                                    </span> is working on this task
                                </div>
                                {entry.description && (
                                    <div className="mt-2 text-sm text-gray-700 bg-gray-50 p-2 rounded border border-gray-200">
                                        <div className="font-medium mb-1">Task Description:</div>
                                        {entry.description}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );

            case "task_complete":
                return (
                    <div key={index} className="p-3 border-b border-gray-200 bg-green-50">
                        <div className="flex items-start">
                            <div className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium mr-2">
                                {timeString}
                            </div>
                            <div>
                                <span className="font-semibold">Task Completed:</span> {entry.task ? getFriendlyTaskName(entry.task) : "Unknown Task"}
                                {entry.content && (
                                    <div className="mt-2 text-sm text-gray-700 bg-white p-2 rounded border border-gray-200 whitespace-pre-wrap">
                                        <div className="font-medium mb-1">Result:</div>
                                        {entry.content}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );

            case "discussion":
                return (
                    <div key={index} className="p-3 border-b border-gray-200 bg-yellow-50">
                        <div className="flex items-start">
                            <div className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-xs font-medium mr-2">
                                {timeString}
                            </div>
                            <div>
                                <span className={`font-semibold ${agentColor}`}>
                                    {entry.agent?.replace("_", " ")}
                                </span> <span className="font-medium">contributed to the discussion:</span>
                                {entry.content && (
                                    <div className="mt-2 text-sm text-gray-700 bg-white p-2 rounded border border-gray-200 whitespace-pre-wrap">
                                        {entry.content}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );

            case "vote_recorded":
                return (
                    <div key={index} className="p-3 border-b border-gray-200 bg-purple-50">
                        <div className="flex items-start">
                            <div className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs font-medium mr-2">
                                {timeString}
                            </div>
                            <div>
                                <span className="font-semibold">Vote Recorded:</span>
                                <div className="mt-1">
                                    <span className={`font-medium ${agentColor}`}>{entry.agent}</span> has voted
                                </div>
                                <div className="grid grid-cols-2 gap-4 mt-2">
                                    <div className="bg-white p-2 rounded border border-gray-200">
                                        <div className="text-sm font-medium mb-1">Interest Rate Vote:</div>
                                        <div className="text-lg font-bold">{entry.vote}</div>
                                    </div>
                                    <div className="bg-white p-2 rounded border border-gray-200">
                                        <div className="text-sm font-medium mb-1">2025 Prediction:</div>
                                        <div className="text-lg font-bold">{entry.prediction}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                );

            case "statement_created":
                return (
                    <div key={index} className="p-3 border-b border-gray-200 bg-blue-50">
                        <div className="flex items-start">
                            <div className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium mr-2">
                                {timeString}
                            </div>
                            <div>
                                <span className="font-semibold">FOMC Statement Created</span>
                                {entry.content && (
                                    <div className="mt-2 text-sm text-gray-700 bg-white p-2 rounded border border-gray-200">
                                        <div className="font-medium mb-1">Preview:</div>
                                        {entry.content}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );

            default:
                return (
                    <div key={index} className="p-3 border-b border-gray-200">
                        <div className="flex items-start">
                            <div className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs font-medium mr-2">
                                {timeString}
                            </div>
                            <div>
                                <span className="font-semibold">System:</span> {entry.type}
                                {entry.content && (
                                    <div className="mt-1 text-sm text-gray-700">
                                        {entry.content}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                );
        }
    };

    return (
        <div className="bg-white rounded-lg shadow-md overflow-hidden h-full flex flex-col">
            <div className="bg-gray-800 text-white p-3">
                <h3 className="text-lg font-medium">Simulation Conversation Log</h3>
                <p className="text-sm text-gray-300">Real-time agent conversations and activities</p>
            </div>

            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto"
                style={{ maxHeight: "600px" }}
            >
                {logs.length === 0 ? (
                    <div className="p-6 text-center text-gray-500">
                        <svg className="w-12 h-12 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                        </svg>
                        <p className="text-lg">No conversation logs yet.</p>
                        <p className="text-sm">Start the simulation to see agent interactions.</p>
                    </div>
                ) : (
                    logs.map((entry, index) => renderLogEntry(entry, index))
                )}
            </div>
        </div>
    );
};

export default ConversationLog;