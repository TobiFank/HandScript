import {Page, pageApi, Writer} from "@/services/api";
import {Card, CardContent} from "@/components/ui/card";
import {Button} from "@/components/ui/button";
import {Edit2, RefreshCw, Trash2, ZoomIn} from 'lucide-react';
import {getStorageUrl} from '@/config/api';
import {useState} from "react";
import {useMutation, useQuery, useQueryClient} from "@tanstack/react-query";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle
} from '@/components/ui/alert-dialog';
import ImagePreviewModal from '@/components/ImagePreviewModal';
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue,} from "@/components/ui/select";

interface PageCardProps {
    page: Page;
    writers: Writer[];
    onEdit: () => void;
}

export function PageCard({page, writers, onEdit}: PageCardProps) {
    const queryClient = useQueryClient();
    const [showDeleteDialog, setShowDeleteDialog] = useState(false);
    const [showPreviewImage, setShowPreviewImage] = useState(false);

    const {data: pageStatus} = useQuery({
        queryKey: ['page-status', page.id],
        queryFn: async () => {
            const response = await pageApi.get(page.id);
            return response.data;
        },
        enabled: page.processing_status === 'processing',
        refetchInterval: page.processing_status === 'processing' ? 2000 : false,
    });

    const currentStatus = pageStatus?.processing_status || page.processing_status;

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed':
                return 'bg-green-100 text-green-700';
            case 'processing':
                return 'bg-blue-100 text-blue-700';
            case 'error':
                return 'bg-red-100 text-red-700';
            default:
                return 'bg-gray-100 text-gray-700';
        }
    };

    const deletePage = useMutation({
        mutationFn: async (pageId: number) => {
            const response = await pageApi.delete(pageId);
            return response;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['document']});
        },
    });

    const reprocessPage = useMutation({
        mutationFn: async (pageId: number) => {
            try {
                const imageResponse = await fetch(getStorageUrl(page?.image_path || ''));
                const imageBlob = await imageResponse.blob();
                const imageFile = new File([imageBlob], `page_${page.id}.jpg`, {type: 'image/jpeg'});
                const formData = new FormData();
                formData.append('files', imageFile);
                await pageApi.delete(pageId);
                const response = await pageApi.upload(page.document_id, formData, page.writer_id);

                // Keep checking status until processing is complete
                const checkStatus = async (): Promise<void> => {
                    const pageResponse = await pageApi.get(pageId);
                    if (pageResponse.data.processing_status === 'processing') {
                        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
                        return checkStatus(); // Recurse until complete
                    }
                    return;
                };

                // Wait for processing to complete before resolving the mutation
                await checkStatus();
                return response;
            } catch (error) {
                console.error('Error during reprocessing:', error);
                throw error;
            }
        },
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['document', page.document_id]});
        },
    });

    const updateWriter = useMutation({
        mutationFn: async ({pageId, writerId}: { pageId: number; writerId: number }) => {
            return pageApi.assignWriter(pageId, writerId);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['document']});
        },
    });

    return (
        <>
            <Card
                className="hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => onEdit()}
            >
                <CardContent className="p-4">
                    {/* Main container - Changed from flex to grid for better layout control */}
                    <div className="grid grid-cols-[auto,1fr] gap-4">
                        {/* Image Container - Now in its own column */}
                        <div className="relative group w-48 h-64 bg-gray-100 rounded-lg overflow-hidden flex-shrink-0">
                            <img
                                src={getStorageUrl(page.image_path)}
                                alt={`Page ${page.page_number}`}
                                className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-200"
                            />
                            <Button
                                variant="secondary"
                                size="sm"
                                className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setShowPreviewImage(true);
                                }}
                            >
                                <ZoomIn className="w-4 h-4 mr-1"/>
                                View
                            </Button>
                        </div>

                        {/* Content Column */}
                        <div className="min-w-0">
                            {/* Top row with metadata and buttons */}
                            <div className="flex justify-between items-start mb-4">
                                {/* Metadata */}
                                <div className="flex items-center">
                                    <span className="text-sm font-medium">Page {page.page_number}</span>
                                    <span className="mx-2 text-gray-300">|</span>
                                    <span className="text-sm text-gray-500">
                        <Select
                            defaultValue={page.writer_id?.toString()}
                            onValueChange={(value) => {
                                updateWriter.mutate({
                                    pageId: page.id,
                                    writerId: parseInt(value)
                                });
                            }}
                        >
    <SelectTrigger className="w-[180px] h-8 text-sm">
        <SelectValue placeholder="Select writer"/>
    </SelectTrigger>
    <SelectContent>
        {writers.map((writer) => (
            <SelectItem
                key={writer.id}
                value={writer.id.toString()}
            >
                {writer.name}
            </SelectItem>
        ))}
    </SelectContent>
</Select>
                    </span>
                                    <span className="mx-2 text-gray-300">|</span>
                                    <span
                                        className={`text-xs px-2 py-0.5 rounded ${getStatusColor(page.processing_status)}`}>
                        {page.processing_status.charAt(0).toUpperCase() + page.processing_status.slice(1)}
                    </span>
                                </div>

                                {/* Buttons */}
                                <div className="flex space-x-2 ml-4">
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            reprocessPage.mutate(page.id);
                                        }}
                                        disabled={reprocessPage.isPending || currentStatus === 'processing'}
                                    >
                                        <RefreshCw
                                            className={`w-4 h-4 mr-1 ${reprocessPage.isPending || currentStatus === 'processing' ? 'animate-spin' : ''}`}/>
                                        {reprocessPage.isPending || currentStatus === 'processing' ? 'Processing...' : 'Re-OCR'}
                                    </Button>

                                    <Button variant="ghost" size="sm" onClick={onEdit}>
                                        <Edit2 className="w-4 h-4 mr-1"/>
                                        Edit
                                    </Button>

                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setShowDeleteDialog(true);
                                        }}
                                    >
                                        <Trash2 className="w-4 h-4 mr-2"/>
                                        Delete
                                    </Button>
                                </div>
                            </div>

                            {/* Text content - Now spans full width under buttons */}
                            <p className="text-sm text-gray-600 break-words">
                                {page.formatted_text || page.extracted_text || 'No text extracted yet'}
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>

            <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Page</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete this page? This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={() => deletePage.mutate(page.id)}
                            className="bg-red-600 hover:bg-red-700"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {/* Image Preview Modal */}
            <ImagePreviewModal
                open={showPreviewImage}
                onOpenChange={setShowPreviewImage}
                imageSrc={getStorageUrl(page.image_path)}
            />
        </>
    );
}