// components/FileUpload.tsx
"use client";

import React, { useState, useRef } from 'react';

export default function FileUpload() {
    const [files, setFiles] = useState<File[]>([]);
    const [datePrefix, setDatePrefix] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [debugInfo, setDebugInfo] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFiles(Array.from(e.target.files));
            // Clear previous messages when files change
            setMessage('');
            setError('');
            setDebugInfo(null);
        }
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            setFiles(Array.from(e.dataTransfer.files));
            // Clear previous messages when files change
            setMessage('');
            setError('');
            setDebugInfo(null);
        }
    };

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleUpload = async () => {
        if (files.length === 0) {
            setError('Please select at least one file to upload');
            return;
        }

        if (!datePrefix) {
            setError('Please enter a date prefix (e.g. "23_6")');
            return;
        }

        setIsUploading(true);
        setError('');
        setMessage('');
        setDebugInfo(null);

        try {
            const formData = new FormData();
            formData.append('date_prefix', datePrefix);

            files.forEach(file => {
                formData.append('files', file);
            });

            // Log what we're sending
            const fileNames = files.map(f => f.name).join(', ');
            console.log(`Uploading files: ${fileNames} with date prefix: ${datePrefix}`);

            const response = await fetch('http://localhost:8000/api/upload', {
                method: 'POST',
                body: formData,
            });

            // Get the full response text for debugging
            const responseText = await response.text();

            let result;
            try {
                // Try to parse as JSON
                result = JSON.parse(responseText);
            } catch (parseError) {
                // If parsing fails, show the raw response
                setDebugInfo(`Server returned non-JSON response: ${responseText}`);
                throw new Error('Server returned invalid JSON');
            }

            if (result.success) {
                setMessage(`Successfully uploaded ${result.files.length} files`);
                setFiles([]);
                setDatePrefix('');

                // Optional: Clear the file input
                if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                }
            } else {
                setError(result.message || 'Upload failed');
                setDebugInfo(`Server response: ${JSON.stringify(result, null, 2)}`);
            }
        } catch (err) {
            console.error('Error uploading files:', err);
            setError('Error uploading files. Please try again.');
            if (err instanceof Error) {
                setDebugInfo(`Error details: ${err.message}`);
            }
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Upload FOMC Data Files</h2>

            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Date Prefix (e.g. "23_6")
                </label>
                <input
                    type="text"
                    value={datePrefix}
                    onChange={(e) => setDatePrefix(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded-md"
                    placeholder="Enter date prefix"
                />
                <p className="mt-1 text-xs text-gray-500">
                    This prefix will be added to PDF and CSV files (e.g. "23_6 beige book.pdf")
                </p>
            </div>

            <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Files
                </label>
                <div
                    className="border-2 border-dashed border-gray-300 rounded-md p-6 text-center hover:bg-gray-50 transition-colors cursor-pointer"
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                    <p className="mt-1 text-sm text-gray-600">
                        Drag and drop files here, or click to select files
                    </p>
                    <p className="mt-1 text-xs text-gray-500">
                        You can select multiple files at once (PDF for reports and CSV for data)
                    </p>
                    <input
                        ref={fileInputRef}
                        id="file-input"
                        type="file"
                        onChange={handleFileChange}
                        className="hidden"
                        multiple
                        accept=".pdf,.csv"
                    />
                </div>
            </div>

            {files.length > 0 && (
                <div className="mb-4">
                    <h3 className="text-sm font-medium mb-2">Selected Files:</h3>
                    <ul className="pl-5 list-disc">
                        {files.map((file, index) => (
                            <li key={index} className="text-sm">
                                {file.name} ({(file.size / 1024).toFixed(2)} KB)
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {error && (
                <div className="mb-4 p-2 bg-red-100 border border-red-400 text-red-700 rounded">
                    {error}
                </div>
            )}

            {message && (
                <div className="mb-4 p-2 bg-green-100 border border-green-400 text-green-700 rounded">
                    {message}
                </div>
            )}

            {debugInfo && (
                <div className="mb-4 p-2 bg-gray-100 border border-gray-400 text-gray-700 rounded text-xs font-mono overflow-auto max-h-40">
                    <details>
                        <summary className="cursor-pointer font-semibold">Debug Information</summary>
                        <pre>{debugInfo}</pre>
                    </details>
                </div>
            )}

            <button
                onClick={handleUpload}
                disabled={isUploading}
                className={`w-full p-2 ${isUploading
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                    } rounded`}
            >
                {isUploading ? 'Uploading...' : 'Upload Files'}
            </button>
        </div>
    );
}