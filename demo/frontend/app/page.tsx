// app/page.tsx
"use client"

import { WebSocketProvider } from '../components/WebSocketProvider';
import FOMCDashboard from '../components/FOMCDashboard';

export default function Home() {
    return (
        <WebSocketProvider>
            <main className="flex min-h-screen flex-col items-center p-4 md:p-8">
                <header className="w-full max-w-7xl mb-8">
                    <h1 className="text-3xl font-bold mb-2">FOMC Simulation Dashboard</h1>
                    <p className="text-gray-600">
                        Multi-Agentic Framework for FOMC Interest Rate Prediction and Interpretability
                    </p>
                </header>

                <div className="w-full max-w-7xl">
                    <FOMCDashboard />
                </div>

                <footer className="mt-12 text-center text-gray-500 text-sm">
                    <p>
                        FOMC Simulation powered by CrewAI - {new Date().getFullYear()}, Made for BNY
                    </p>
                </footer>
            </main>
        </WebSocketProvider>
    );
}