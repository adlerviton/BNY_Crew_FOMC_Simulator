/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    swcMinify: true,
    webpack: (config) => {
        // Handle chart.js imports
        config.resolve.fallback = {
            ...config.resolve.fallback,
            canvas: false
        };
        return config;
    },
}

module.exports = nextConfig