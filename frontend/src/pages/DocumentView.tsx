// src/pages/DocumentView.tsx
import {useNavigate, useParams} from 'react-router-dom';
import {useMutation, useQuery, useQueryClient} from '@tanstack/react-query';
import {ChevronRight, Eye, Trash2, Upload} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {documentApi, pageApi, projectApi, writerApi,} from '@/services/api';
import UploadPagesDialog from '@/components/dialogs/UploadPagesDialog';
import {useState} from 'react';
import {PageCard} from '@/components/PageCard';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';


export default function DocumentView() {
    const {documentId} = useParams();
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [showUpload, setShowUpload] = useState(false);
    const [showDeleteDialog, setShowDeleteDialog] = useState(false);
    const [processingPages, setProcessingPages] = useState(false);

    // Fetch document data
    const { data: document, isLoading: isDocumentLoading } = useQuery({
        queryKey: ['document', documentId],
        queryFn: async () => {
            if (!documentId) throw new Error('No document ID');
            const response = await documentApi.getOne(Number(documentId));
            // Sort pages by page_number to ensure correct order
            return {
                ...response.data,
                pages: response.data.pages?.sort((a, b) => a.page_number - b.page_number)
            };
        },
        enabled: !!documentId,
    });

    // Fetch project data
    const {data: project} = useQuery({
        queryKey: ['project', document?.project_id],
        queryFn: async () => {
            if (!document?.project_id) throw new Error('No project ID');
            const response = await projectApi.get(document.project_id);
            return response.data;
        },
        enabled: !!document?.project_id,
    });

    // Delete mutation
    const deleteDocument = useMutation({
        mutationFn: async () => {
            if (!documentId) throw new Error('No document ID');
            await documentApi.delete(Number(documentId));
        },
        onSuccess: () => {
            // Navigate back to project view and invalidate queries
            if (document?.project_id) {
                queryClient.invalidateQueries({queryKey: ['documents', document.project_id]});
                navigate(`/projects/${document.project_id}`);
            } else {
                navigate('/projects');
            }
        },
    });

    // Fetch writers for the PageCard component
    const {data: writers = []} = useQuery({
        queryKey: ['writers'],
        queryFn: async () => {
            const response = await writerApi.list();
            return response.data;
        },
    });

    // Handle page upload
    const uploadPages = useMutation({
        mutationFn: async ({files, writerId}: { files: File[]; writerId: number }) => {
            setProcessingPages(true);
            const formData = new FormData();
            files.forEach((file) => {
                formData.append('files', file);
            });
            const response = await pageApi.upload(Number(documentId), formData, writerId);

            // Keep checking status until all pages are processed
            const checkStatus = async (): Promise<void> => {
                const docResponse = await documentApi.getOne(Number(documentId));
                const newPages = docResponse.data.pages || [];
                const stillProcessing = newPages.some(
                    page => page.processing_status === 'processing'
                );

                if (stillProcessing) {
                    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
                    return checkStatus();
                }
                return;
            };

            // Wait for processing to complete
            await checkStatus();
            return response.data;
        },
        onSuccess: () => {
            setProcessingPages(false);
            queryClient.invalidateQueries({queryKey: ['document', documentId]});
            setShowUpload(false);
        },
        onError: () => {
            setProcessingPages(false);
        }
    });

    const reorderPage = useMutation({
        mutationFn: async ({ pageId, newPosition }: { pageId: number; newPosition: number }) => {
            return pageApi.reorder(pageId, newPosition);
        },
        onSuccess: () => {
            // Invalidate and refetch document to get updated page order
            queryClient.invalidateQueries({ queryKey: ['document', documentId] });
        },
    });

    const handleDragEnd = (result: any) => {
        if (!result.destination || !document?.pages) return;

        const oldIndex = result.source.index;
        const newIndex = result.destination.index;

        if (oldIndex === newIndex) return;

        const pageId = document.pages[oldIndex].id;
        const newPosition = newIndex + 1; // Convert to 1-based page numbers

        // Optimistically update the UI
        const newPages = Array.from(document.pages);
        const [movedPage] = newPages.splice(oldIndex, 1);
        newPages.splice(newIndex, 0, movedPage);

        // Update the cache immediately for smooth UI
        queryClient.setQueryData(['document', documentId], (old: any) => ({
            ...old,
            pages: newPages.map((page, idx) => ({
                ...page,
                page_number: idx + 1
            }))
        }));

        // Then send to server
        reorderPage.mutate({ pageId, newPosition });
    };

    if (isDocumentLoading) {
        return (
            <div className="p-8">
                <div className="max-w-5xl mx-auto">
                    <div className="animate-pulse">
                        <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
                        <div className="h-4 bg-gray-200 rounded w-1/2 mb-8"></div>
                        <div className="space-y-4">
                            <div className="h-32 bg-gray-200 rounded"></div>
                            <div className="h-32 bg-gray-200 rounded"></div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    if (!document) {
        return (
            <div className="p-8">
                <div className="max-w-5xl mx-auto">
                    <h1 className="text-2xl font-bold text-gray-900 mb-2">Document not found</h1>
                    <Button onClick={() => navigate('/projects')}>Back to Projects</Button>
                </div>
            </div>
        );
    }

    return (
        <div className="p-8">
            <div className="max-w-5xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                    <div className="flex items-center text-sm text-gray-500 mb-2">
                        <span>{project?.name}</span>
                        <ChevronRight className="w-4 h-4 mx-2"/>
                        <span>{document.name}</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold">{document.name}</h1>
                            <p className="text-gray-500">{document.description}</p>
                        </div>
                        <div className="flex space-x-3">
                            <Button
                                variant="outline"
                                onClick={() => navigate(`/documents/${documentId}/preview`)}
                            >
                                <Eye className="w-4 h-4 mr-2"/>
                                Preview
                            </Button>
                            <Button onClick={() => setShowUpload(true)}>
                                <Upload className="w-4 h-4 mr-2"/>
                                Add Pages
                            </Button>
                            <Button
                                variant="outline"
                                onClick={() => setShowDeleteDialog(true)}
                                className="text-red-600 hover:text-red-700 hover:bg-red-50"
                            >
                                <Trash2 className="w-4 h-4 mr-2"/>
                                Delete
                            </Button>
                        </div>
                    </div>
                </div>

                {/* Pages List */}
                <div className="space-y-4">
                    <DragDropContext onDragEnd={handleDragEnd}>
                        <Droppable droppableId="pages">
                            {(provided) => (
                                <div
                                    {...provided.droppableProps}
                                    ref={provided.innerRef}
                                    className="space-y-4"
                                >
                                    {document.pages?.map((page, index) => (
                                        <Draggable
                                            key={page.id}
                                            draggableId={String(page.id)}
                                            index={index}
                                        >
                                            {(provided) => (
                                                <div
                                                    ref={provided.innerRef}
                                                    {...provided.draggableProps}
                                                    {...provided.dragHandleProps}
                                                >
                                                    <PageCard
                                                        page={page}
                                                        writers={writers}
                                                        onEdit={() => navigate(`/pages/${page.id}/edit`)}
                                                    />
                                                </div>
                                            )}
                                        </Draggable>
                                    ))}
                                    {provided.placeholder}
                                </div>
                            )}
                        </Droppable>
                    </DragDropContext>

                    {/* Upload Card */}
                    <div
                        className="border-2 border-dashed border-gray-200 rounded-lg p-8 text-center hover:border-blue-400 hover:bg-blue-50/50 transition-colors cursor-pointer"
                        onClick={() => setShowUpload(true)}
                    >
                        <Upload className="w-6 h-6 text-gray-400 mx-auto mb-2"/>
                        <p className="text-sm text-gray-500">
                            Drop images here or click to upload
                        </p>
                    </div>
                </div>
            </div>

            <UploadPagesDialog
                open={showUpload}
                onOpenChange={(open) => {
                    // Prevent closing while processing
                    if (!uploadPages.isPending && !processingPages) {
                        setShowUpload(open);
                    }
                }}
                onUpload={(files, writerId) => uploadPages.mutate({files, writerId})}
                isUploading={uploadPages.isPending || processingPages}
            />
            {/* Delete Confirmation Dialog */}
            <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Document</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete this document? This action cannot be undone.
                            All pages within this document will be permanently deleted.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={() => deleteDocument.mutate()}
                            className="bg-red-600 hover:bg-red-700"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
}