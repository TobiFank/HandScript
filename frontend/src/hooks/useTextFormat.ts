// src/hooks/useTextFormat.ts
import { useCallback } from 'react';

export function useTextFormat() {
    const applyFormat = useCallback((command: string, value?: string) => {
        document.execCommand(command, false, value);
    }, []);

    const formatText = useCallback((format: string) => {
        switch (format) {
            case 'bold':
                applyFormat('bold');
                break;
            case 'italic':
                applyFormat('italic');
                break;
            case 'underline':
                applyFormat('underline');
                break;
            case 'alignLeft':
                applyFormat('justifyLeft');
                break;
            case 'alignCenter':
                applyFormat('justifyCenter');
                break;
            case 'alignRight':
                applyFormat('justifyRight');
                break;
            case 'list':
                applyFormat('insertUnorderedList');
                break;
            default:
                console.warn('Unknown format:', format);
        }
    }, [applyFormat]);

    return { formatText };
}