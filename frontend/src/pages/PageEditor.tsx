import {useNavigate, useParams} from 'react-router-dom';
import {useMutation, useQuery, useQueryClient} from '@tanstack/react-query';
import {ChevronLeft, ChevronRight, GraduationCap, RotateCcw, Save, ZoomIn, ZoomOut} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {Tooltip, TooltipContent, TooltipTrigger} from '@/components/ui/tooltip';
import {documentApi, LineSegment, pageApi, writerApi} from '@/services/api';
import {useEffect, useState} from 'react';
import FormatToolbar from '@/components/editor/FormatToolbar';
import {getStorageUrl} from '@/config/api';
import {toast} from "sonner";

interface LineEdit {
    image?: string;
    text: string;
    bbox?: [number, number, number, number];
}

export default function PageEditor() {
    const {pageId} = useParams();
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [zoomLevel, setZoomLevel] = useState(100);
    const [lineEdits, setLineEdits] = useState<LineEdit[]>([]);
    const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
    const [selectedLine, setSelectedLine] = useState<number | null>(null);
    const [, setImageDimensions] = useState<{ width: number, height: number }>({width: 0, height: 0});

    // Fetch page data
    const {data: pageData, isLoading: isPageLoading} = useQuery({
        queryKey: ['page', pageId],
        queryFn: async () => {
            const response = await pageApi.get(Number(pageId));
            return response.data;
        }
    });

    // Fetch document data
    const {data: documentData, isLoading: isDocumentLoading} = useQuery({
        queryKey: ['document', pageData?.document_id],
        queryFn: async () => {
            if (!pageData?.document_id) return null;
            const response = await documentApi.getOne(pageData.document_id);
            return response.data;
        },
        enabled: !!pageData?.document_id
    });

    // Initialize line edits when page loads
    useEffect(() => {
        if (pageData && Array.isArray(pageData.lines) && pageData.lines.length > 0) {
            setLineEdits(pageData.lines.map((line: LineSegment) => ({
                image: line.image_path,
                text: line.text,
                bbox: line.bbox
            })));
        } else if (pageData?.formatted_text) {
            // Fallback if no line data
            setLineEdits([{text: pageData.formatted_text}]);
        }
    }, [pageData]);

    // Save handler
    const saveMutation = useMutation({
        mutationFn: async () => {
            const updatedLines = lineEdits.map(line => ({
                text: line.text,
                image_path: line.image,
                bbox: line.bbox,
            }));

            return pageApi.update(Number(pageId), {
                formatted_text: lineEdits.map(line => line.text).join('\n'),
                lines: updatedLines
            });
        },
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['page', pageId]});
            if (pageData?.document_id) {
                queryClient.invalidateQueries({queryKey: ['document', pageData.document_id]});
            }
            setHasUnsavedChanges(false);
            toast.success('Changes saved successfully');
        }
    });

    // Handle training sample creation
    const addToTraining = useMutation({
        mutationFn: async () => {
            if (!pageData?.writer_id || !lineEdits || lineEdits.length === 0) {
                throw new Error('No writer assigned or no line data available');
            }

            // Submit each existing line as a training sample
            const promises = lineEdits.map(async (line) => {
                // Only use lines that have both image and text
                if (!line.image) return null;

                try {
                    const imageResponse = await fetch(getStorageUrl(line.image));
                    const imageBlob = await imageResponse.blob();
                    const imageFile = new File([imageBlob],
                        `line_${Math.random()}.png`,
                        {type: 'image/jpeg'}
                    );

                    // Just submit the image and text directly
                    return writerApi.addTrainingSample(
                        pageData.writer_id!,
                        imageFile,
                        line.text
                    );
                } catch (error) {
                    console.error('Error creating training sample:', error);
                    return null;
                }
            });

            return Promise.all(promises);
        },
        onSuccess: () => {
            toast.success('Added lines to training samples');
            if (pageData?.writer_id) {
                queryClient.invalidateQueries({
                    queryKey: ['training-samples', pageData.writer_id]
                });
            }
        },
        onError: (error) => {
            toast.error('Failed to add training samples');
            console.error('Error adding training samples:', error);
        }
    });

    // Navigation between pages
    const navigateToPage = (direction: 'prev' | 'next') => {
        if (!pageData?.document_id) return;

        const newPageNumber = direction === 'prev'
            ? pageData.page_number - 1
            : pageData.page_number + 1;

        navigate(`/documents/${pageData.document_id}/pages/${newPageNumber}`);
    };

    if (isPageLoading || isDocumentLoading) {
        return <div className="p-8">Loading page...</div>;
    }

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Editor Header */}
            <div className="h-16 bg-white border-b flex items-center justify-between px-6">
                <div className="flex items-center space-x-4">
                    <button
                        className="text-gray-600 hover:text-gray-900"
                        onClick={() => pageData?.document_id &&
                            navigate(`/documents/${pageData.document_id}`)}
                    >
                        <ChevronLeft className="w-6 h-6"/>
                    </button>
                    <div>
                        <h1 className="font-medium">Page {pageData?.page_number}</h1>
                        <p className="text-sm text-gray-500">
                            {documentData?.name}
                        </p>
                    </div>
                </div>

                <div className="flex items-center space-x-3">
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="ghost"
                                onClick={() => {
                                    if (pageData?.formatted_text) {
                                        setLineEdits([{text: pageData.formatted_text}]);
                                    }
                                    setHasUnsavedChanges(false);
                                }}
                                disabled={!hasUnsavedChanges}
                            >
                                <RotateCcw className="w-4 h-4"/>
                                <span className="ml-2">Reset</span>
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                            Reset to original text
                        </TooltipContent>
                    </Tooltip>

                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="outline"
                                onClick={() => addToTraining.mutate()}
                                disabled={!pageData?.writer_id || addToTraining.isPending}
                            >
                                <GraduationCap className="w-4 h-4 mr-2"/>
                                {addToTraining.isPending ? 'Adding...' : 'Add to Training'}
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                            {pageData?.writer_id
                                ? 'Add these lines as training samples'
                                : 'Assign a writer first to add as training sample'}
                        </TooltipContent>
                    </Tooltip>

                    <Button
                        onClick={() => saveMutation.mutate()}
                        disabled={!hasUnsavedChanges || saveMutation.isPending}
                    >
                        <Save className="w-4 h-4 mr-2"/>
                        {saveMutation.isPending ? 'Saving...' : 'Save Changes'}
                    </Button>
                </div>
            </div>

            {/* Editor Content */}
            <div className="flex-1 flex">
                {/* Original Image Side */}
                <div className="w-1/2 p-4 border-r border-gray-200">
                    <div className="bg-white rounded-lg shadow-sm p-4 h-full flex flex-col">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="font-medium">Original</h2>
                            <div className="flex items-center space-x-2">
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => setZoomLevel(Math.max(25, zoomLevel - 25))}
                                            disabled={zoomLevel <= 25}
                                        >
                                            <ZoomOut className="w-4 h-4"/>
                                        </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Zoom Out</TooltipContent>
                                </Tooltip>
                                <span className="text-sm text-gray-600">{zoomLevel}%</span>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => setZoomLevel(Math.min(200, zoomLevel + 25))}
                                            disabled={zoomLevel >= 200}
                                        >
                                            <ZoomIn className="w-4 h-4"/>
                                        </Button>
                                    </TooltipTrigger>
                                    <TooltipContent>Zoom In</TooltipContent>
                                </Tooltip>
                            </div>
                        </div>

                        <div className="flex-1 overflow-auto bg-gray-50 rounded border">
                            <div className="relative">
                                <img
                                    src={getStorageUrl(pageData?.image_path || '')}
                                    alt="Original page"
                                    className="w-full"
                                    style={{
                                        transform: `scale(${zoomLevel / 100})`,
                                        transformOrigin: 'top left'
                                    }}
                                    onLoad={(e) => {
                                        const img = e.target as HTMLImageElement;
                                        setImageDimensions({
                                            width: img.naturalWidth,
                                            height: img.naturalHeight
                                        });
                                    }}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Text Editor Side */}
                <div className="w-1/2 p-4">
                    <div className="bg-white rounded-lg shadow-sm p-4 h-full flex flex-col">
                        <FormatToolbar onFormatClick={(format) => {
                            // Implement format handling
                            console.log('Format clicked:', format);
                        }}/>

                        <div className="flex-1 overflow-auto">
                            {lineEdits.map((line, index) => (
                                <div
                                    key={index}
                                    className={`p-4 border rounded mb-4 ${
                                        selectedLine === index ? 'ring-2 ring-blue-500' : ''
                                    }`}
                                    onClick={() => setSelectedLine(index)}
                                >
                                    {line.image && (
                                        <div className="mb-2">
                                            <img
                                                src={getStorageUrl(line.image)}
                                                alt={`Line ${index + 1}`}
                                                className="max-h-24 object-contain"
                                            />
                                        </div>
                                    )}
                                    <div
                                        ref={(el) => {
                                            // If this is first render and we have text, set initial content
                                            if (el && !el.innerHTML && line.text) {
                                                el.innerHTML = line.text;
                                            }
                                        }}
                                        className="prose max-w-none focus:outline-none focus:ring-2 focus:ring-blue-500 rounded p-2"
                                        contentEditable
                                        suppressContentEditableWarning
                                        onInput={(e) => {
                                            const newLineEdits = [...lineEdits];
                                            newLineEdits[index] = {
                                                ...newLineEdits[index],
                                                text: e.currentTarget.innerHTML
                                            };
                                            setLineEdits(newLineEdits);
                                            setHasUnsavedChanges(true);
                                        }}
                                    />
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Page Navigation */}
            <div className="h-12 bg-white border-t flex items-center justify-between px-6">
                <Button
                    variant="ghost"
                    onClick={() => navigateToPage('prev')}
                    disabled={pageData?.page_number === 1}
                >
                    <ChevronLeft className="w-4 h-4 mr-2"/>
                    Previous Page
                </Button>
                <Button
                    variant="ghost"
                    onClick={() => navigateToPage('next')}
                >
                    Next Page
                    <ChevronRight className="w-4 h-4 ml-2"/>
                </Button>
            </div>
        </div>
    );
}