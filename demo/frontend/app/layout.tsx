// app/layout.tsx
import './globals.css';

export const metadata = {
    title: 'FOMC Simulation Dashboard',
    description: 'Real-time visualization of the Federal Open Market Committee simulation',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <head>
                <link rel="icon" href="/favicon.ico" />
            </head>
            <body>
                {children}
            </body>
        </html>
    );
}