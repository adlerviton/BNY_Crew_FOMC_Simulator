// frontend/components/PredictionChart.tsx
"use client"

import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { PredictionData, memberColors } from '../lib/types';

interface PredictionChartProps {
    data: PredictionData[];
}

const PredictionChart: React.FC<PredictionChartProps> = ({ data }) => {
    // Transform the data for the chart
    const chartData = useMemo(() => {
        return data.map(item => {
            // Convert percentage string to number
            const predictionValue = parseFloat(item.prediction.replace('%', ''));

            return {
                name: item.member,
                value: predictionValue,
                fill: memberColors[item.member] || '#999',
            };
        });
    }, [data]);

    const formatYAxis = (value: number) => `${value}%`;

    return (
        <div className="w-full h-full min-h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                    data={chartData}
                    margin={{
                        top: 20,
                        right: 30,
                        left: 20,
                        bottom: 50,
                    }}
                >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                        dataKey="name"
                        angle={-45}
                        textAnchor="end"
                        height={80}
                        tick={{ fontSize: 12 }}
                    />
                    <YAxis
                        tickFormatter={formatYAxis}
                        domain={[0, 'auto']} // Dynamically adjust based on data
                    />
                    <Tooltip
                        formatter={(value: number) => [`${value}%`, 'Rate Prediction for Next Year']}
                        labelFormatter={(label) => `Member: ${label}`}
                    />
                    <Legend />
                    <Bar
                        dataKey="value"
                        name="Rate Prediction (%)"
                        isAnimationActive={true}
                        animationDuration={500}
                        animationEasing="ease-in-out"
                    />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default PredictionChart;