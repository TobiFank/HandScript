// src/services/api.ts

// Base types matching backend schema
import axios from "axios";
import {API_CONFIG} from '@/config/api';


export interface Project {
    id: number;
    name: string;
    description?: string;
    created_at: string;
    document_count?: number; // Added from Frontend Integration Guide
}

export interface Document {
    id: number;
    project_id: number;
    name: string;
    description?: string;
    created_at: string;
    pages?: Page[];
}

export interface Page {
    id: number;
    document_id: number;
    writer_id?: number;
    image_path: string;
    extracted_text?: string;
    formatted_text?: string;
    page_number: number;
    processing_status: 'pending' | 'processing' | 'completed' | 'error';
    created_at: string;
    updated_at: string;
    lines?: LineSegment[];
}

export interface Writer {
    id: number;
    name: string;
    model_path?: string;
    created_at: string;
    status: 'untrained' | 'training' | 'ready' | 'error';
    accuracy?: number;
    pages_processed: number;
    last_trained?: string;
    is_trained: boolean;
    training_samples?: TrainingSample[];
}

export interface LineSegment {
    bbox: [number, number, number, number];
    text: string;
    image_path: string;
    confidence: number;
}

export interface TrainingSample {
    line_segments: boolean;
    id: number;
    writer_id: number;
    image_path: string;
    text: string;
    created_at: string;
    lines?: LineSegment[];  // Optional to maintain compatibility
    line_count?: number;    // Optional to maintain compatibility
    needs_review?: boolean; // Optional to maintain compatibility
}

// Added from Backend API Specification
export interface WriterStats {
    accuracy_trend: Array<{
        date: string;
        accuracy: number;
    }>;
    avg_processing_time: number;
    char_accuracy: number;
    word_accuracy: number;
    total_pages: number;
    error_types: Array<{
        type: string;
        count: number;
    }>;
}

// Create axios instance with config
const api = axios.create(API_CONFIG);

// Add response interceptor for better error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        console.error('API Error:', error.response || error);
        throw error;
    }
);


export const projectApi = {
    list: () => api.get<Project[]>('/projects'),
    create: (data: { name: string; description?: string }) =>
        api.post<Project>('/projects', data),
    get: (id: number) => api.get<Project>(`/projects/${id}`),
    update: (id: number, data: { name: string; description?: string }) =>
        api.put<Project>(`/projects/${id}`, data),
    delete: (id: number) => api.delete(`/projects/${id}`),
};

export const documentApi = {
    // For getting documents list by project ID
    get: (projectId: number) =>
        api.get<Document[]>(`/documents/project/${projectId}`),

    // For getting a single document
    getOne: (id: number) =>
        api.get<Document>(`/documents/${id}`),

    create: (projectId: number, data: { name: string; description?: string }) =>
        api.post<Document>('/documents', {
            ...data,
            project_id: projectId
        }),

    update: (id: number, data: { name: string; description?: string }) =>
        api.put<Document>(`/documents/${id}`, data),

    delete: (id: number) => api.delete(`/documents/${id}`),

    export: (id: number, format: 'pdf' | 'docx') =>
        api.get(`/documents/${id}/export`, {
            params: {format},
            responseType: 'blob',
            headers: {
                'Accept': format === 'pdf' ? 'application/pdf' : 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            }
        }),

    getForExport: (id: number) =>
        api.get<Document>(`/documents/${id}`, {
            params: {
                include_pages: true,
                include_text: true
            }
        })
};

export const pageApi = {
    upload: (documentId: number, formData: FormData, writerId?: number) =>
        api.post<Page[]>(
            `/pages/upload/${documentId}${writerId ? `?writer_id=${writerId}` : ''}`,
            formData,
            {
                headers: {'Content-Type': 'multipart/form-data'},
            }
        ),
    get: (id: number) => api.get<Page>(`/pages/${id}`),
    update: (id: number, data: {
        formatted_text: string;
        lines: { image_path: string | undefined; bbox: [number, number, number, number] | undefined; text: string }[]
    }) => api.put<Page>(`/pages/${id}`, data),
    delete: (id: number) => api.delete(`/pages/${id}`),
    assignWriter: (id: number, writerId: number) =>
        api.put<Page>(`/pages/${id}/writer`, {writer_id: writerId}),
    reprocess: (id: number) =>
        api.post<Page>(`/pages/${id}/reprocess`),
    reorder: (pageId: number, newPosition: number) =>
        api.put<Page>(`/pages/${pageId}/reorder`, {new_position: newPosition}),
};

export const writerApi = {
    list: () => api.get<Writer[]>('/writers'),
    get: (id: number) => api.get<Writer>(`/writers/${id}`),
    delete: (id: number) => api.delete(`/writers/${id}`),
    train: async (id: number, files: File[], texts: string[]) => {
        const formData = new FormData();

        // Add each file to the formData
        files.forEach((file) => {
            formData.append('files', file);
        });

        // Add each text string to the formData
        texts.forEach((text) => {
            formData.append('texts', text);
        });

        // Debug log to verify data
        console.log('Training data:', {
            writerId: id,
            fileCount: files.length,
            textCount: texts.length
        });

        return api.post<{ success: boolean; message: string }>(
            `/writers/${id}/train`,
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            }
        );
    },
    getTrainingSamples: (writerId: number) =>
        api.get<TrainingSample[]>(`/training-samples/writer/${writerId}`),

    addTrainingSample: async (writerId: number, file: File, text: string) => {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('text', text);
        return api.post<TrainingSample>(
            `/training-samples/writer/${writerId}`,
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            }
        );
    },

    newTrainingSample: async (writerId: number, file: File) => {
        const formData = new FormData();
        formData.append('image', file);
        return api.post<TrainingSample>(
            `/training-samples/writer/${writerId}/process`,  // New endpoint
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            }
        );
    },

    deleteTrainingSample: (sampleId: number) =>
        api.delete(`/training-samples/${sampleId}`),

    startTraining: (writerId: number) =>
        api.post<{ success: boolean; message: string }>(
            `/writers/${writerId}/train`
        ),

    // Update create method to not require immediate training
    create: (data: { name: string; language?: string }) =>
        api.post<Writer>('/writers', data),

    getStats: (writerId: number) =>
        api.get<WriterStats>(`/writers/${writerId}/stats`),

    updateTrainingSample: (sampleId: number, text: string) =>
        api.put<TrainingSample>(`/training-samples/${sampleId}`, {text}),
    update: (id: number, data: { name: string; language?: string }) =>
        api.put<Writer>(`/writers/${id}`, data),
};

// Types for component use
export interface PageWithRelations extends Page {
    writer: Writer | null;
}

export interface DocumentWithRelations extends Omit<Document, 'pages'> {
    project: Project | null;
    pages: PageWithRelations[];
}