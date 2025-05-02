// frontend/components/FOMCStatement.tsx
"use client"

import React from 'react';

interface FOMCStatementProps {
    statement: string;
}

const FOMCStatement: React.FC<FOMCStatementProps> = ({ statement }) => {
    return (
        <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="p-4 bg-blue-800 text-white">
                <h3 className="text-lg font-semibold">Federal Open Market Committee Statement</h3>
            </div>
            <div className="p-6 max-h-96 overflow-y-auto">
                {statement ? (
                    <div className="prose max-w-none fomc-statement">
                        {statement.split('\n\n').map((paragraph, idx) => (
                            <p key={idx} className="mb-4">
                                {paragraph}
                            </p>
                        ))}
                    </div>
                ) : (
                    <div className="py-12 flex flex-col items-center justify-center text-gray-400">
                        <svg className="w-12 h-12 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                        </svg>
                        <p className="text-lg">Statement will appear here when the simulation completes.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default FOMCStatement;