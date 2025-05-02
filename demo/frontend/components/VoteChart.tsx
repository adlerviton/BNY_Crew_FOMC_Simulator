// frontend/components/VoteDisplay.tsx
"use client"
import React from 'react';
import { VoteData } from '../lib/types';

interface VoteDisplayProps {
    data: VoteData[];
}

const VoteDisplay: React.FC<VoteDisplayProps> = ({ data }) => {
    // State tracking
    const [currentData, setCurrentData] = React.useState<VoteData[]>(data);

    React.useEffect(() => {
        console.log("Vote data updated:", data); // Debug log
        setCurrentData(data);
    }, [data]);

    // Check if all votes are -999 (waiting)
    const allWaiting = currentData.every(item => {
        try {
            const voteStr = item.vote ? item.vote.replace(/%/g, '') : '-999';
            const voteValue = parseFloat(voteStr);
            return voteValue === -9999;
        } catch (e) {
            console.error("Error parsing vote data:", e);
            return true;
        }
    });

    // If all are waiting (-999), show waiting message
    if (allWaiting) {
        return (
            <div className="w-full">
                <h2 className="text-lg font-bold mb-3">FOMC Committee Votes</h2>
                <div className="p-4 bg-gray-50 border border-gray-200 rounded text-gray-500">
                    Waiting for committee members to cast their votes...
                </div>
            </div>
        );
    }

    // Function to determine vote action and styling
    const getVoteDisplay = (vote: string | undefined) => {
        try {
            // Handle undefined or -999 votes
            if (!vote) return null;

            const voteStr = vote.replace(/%/g, '');
            const voteValue = parseFloat(voteStr);

            // Skip -999 votes
            if (voteValue === -9999) return null;

            if (voteValue === 0) {
                return {
                    action: 'maintain',
                    textColor: '#333333',
                    bgColor: '#f0f0f0'
                };
            } else if (voteValue > 0) {
                return {
                    action: 'hike',
                    textColor: '#c62828',
                    bgColor: '#ffebee'
                };
            } else {
                return {
                    action: 'cut',
                    textColor: '#2e7d32',
                    bgColor: '#e8f5e9'
                };
            }
        } catch (e) {
            console.error("Error determining vote display:", e);
            return null;
        }
    };

    return (
        <div className="w-full">
            <h2 className="text-lg font-bold mb-3">FOMC Committee Votes</h2>
            <div className="space-y-3">
                {currentData.map((item, index) => {
                    const voteDisplay = getVoteDisplay(item.vote);

                    // Skip members who haven't voted yet (-999)
                    if (!voteDisplay) return null;

                    const { action, textColor, bgColor } = voteDisplay;
                    const itemKey = `${item.member}-${item.vote}-${index}`;

                    return (
                        <div
                            key={itemKey}
                            className="p-3 rounded"
                            style={{ backgroundColor: bgColor }}
                        >
                            <div className="font-bold">{item.member}</div>
                            <div>
                                voted to{' '}
                                <span style={{ color: textColor, fontWeight: 'bold' }}>
                                    {action}
                                </span>
                                {' '}rates
                            </div>
                            {item.prediction && item.prediction !== "0.00%" && item.prediction !== "0%" && item.prediction !== "-999%" && (
                                <div className="text-sm text-gray-600 mt-1">
                                    2025 Prediction: {item.prediction}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default VoteDisplay;