export const API_CONFIG = {
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
    timeout: 30000,
    withCredentials: false, // Changed from true since we're not using cookies
    headers: {
        'Content-Type': 'application/json',
    },
} as const;

export function getStorageUrl(path: string): string {
    const baseUrl = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace('/api', '');
    if (!path) return '';
    const cleanBaseUrl = baseUrl.replace(/\/$/, '');
    const cleanPath = path.replace(/^\//, '');
    return `${cleanBaseUrl}/storage/${cleanPath}`;
}