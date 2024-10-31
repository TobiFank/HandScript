// src/hooks/useEditorState.ts
import { useState, useCallback } from 'react';
import { debounce } from 'lodash';

interface EditorState {
    text: string;
    hasUnsavedChanges: boolean;
}

export function useEditorState(initialText: string) {
    const [state, setState] = useState<EditorState>({
        text: initialText,
        hasUnsavedChanges: false,
    });

    const handleTextChange = useCallback(
        debounce((newText: string) => {
            setState({
                text: newText,
                hasUnsavedChanges: true,
            });
        }, 500),
        []
    );

    const resetState = useCallback(() => {
        setState({
            text: initialText,
            hasUnsavedChanges: false,
        });
    }, [initialText]);

    const clearUnsavedChanges = useCallback(() => {
        setState(prev => ({
            ...prev,
            hasUnsavedChanges: false,
        }));
    }, []);

    return {
        text: state.text,
        hasUnsavedChanges: state.hasUnsavedChanges,
        handleTextChange,
        resetState,
        clearUnsavedChanges,
    };
}