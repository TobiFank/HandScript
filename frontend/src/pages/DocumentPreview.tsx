import {useNavigate, useParams} from 'react-router-dom';
import {useQuery} from '@tanstack/react-query';
import {ChevronLeft, Download, X, ZoomIn} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger} from '@/components/ui/dropdown-menu';
import {type Document, documentApi, type Page, type Project, projectApi, type Writer, writerApi} from '@/services/api';
import {useState} from 'react';
import {Dialog, DialogContent} from '@/components/ui/dialog';
import {cn} from '@/lib/utils';
import {getStorageUrl} from '@/config/api';
import ImagePreviewModal from '@/components/ImagePreviewModal';
import {toast} from "sonner";

interface EnhancedPage extends Page {
    writer: Writer | null;
}

interface DocumentWithRelations extends Omit<Document, 'pages'> {
    project: Project | null;
    pages: EnhancedPage[];
}

export default function DocumentPreview() {
    const {documentId} = useParams();
    const navigate = useNavigate();
    const [selectedPage, setSelectedPage] = useState<number | null>(null);
    const [showShareDialog, setShowShareDialog] = useState(false);
    const [previewImage, setPreviewImage] = useState<string | null>(null);

    // Fetch document data
    const {data: documentData} = useQuery({
        queryKey: ['document', documentId],
        queryFn: async () => {
            const response = await documentApi.getOne(Number(documentId));
            return {
                ...response.data,
                pages: response.data.pages ?
                    [...response.data.pages].sort((a, b) => a.page_number - b.page_number)
                    : []
            };
        },
    });

    // Fetch project data
    const {data: projectData} = useQuery({
        queryKey: ['project', documentData?.project_id],
        queryFn: async () => {
            if (!documentData?.project_id) return null;
            const response = await projectApi.get(documentData.project_id);
            return response.data as Project;
        },
        enabled: !!documentData?.project_id,
    });

    // Fetch writers data
    const {data: writers = []} = useQuery({
        queryKey: ['writers'],
        queryFn: async () => {
            const response = await writerApi.list();
            return response.data as Writer[];
        },
    });

    // Combine all data
    const document: DocumentWithRelations | null = documentData ? {
        ...documentData,
        project: projectData || null,
        pages: (documentData.pages || []).map(page => ({
            ...page,
            writer: writers.find(w => w.id === page.writer_id) || null
        }))
    } : null;

    const handleExport = async (format: 'pdf' | 'docx') => {
        if (!document) return;

        try {
            // First ensure we have the complete document with all pages
            const completeDoc = await documentApi.getForExport(document.id);

            // Verify we have all the necessary content
            const hasAllContent = completeDoc.data.pages?.every(
                page => page.formatted_text || page.extracted_text
            );

            if (!hasAllContent) {
                toast.error('Some pages are missing content. Please ensure all pages are processed.');
                return;
            }

            // Proceed with export
            const response = await documentApi.export(document.id, format);

            // Create a download link for the blob
            const url = window.URL.createObjectURL(response.data);
            const a = window.document.createElement('a');
            a.href = url;
            a.download = `${document.name}.${format}`;
            window.document.body.appendChild(a);
            a.click();

            // Cleanup
            window.document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            toast.success(`Document exported as ${format.toUpperCase()} successfully`);
        } catch (error) {
            console.error('Export failed:', error);
            toast.error(`Failed to export document as ${format.toUpperCase()}`);
        }
    };

    if (!document) {
        return <div>Loading...</div>;
    }

    return (
        <div className="flex flex-col h-screen bg-gray-50">
            {/* Preview Header */}
            <div className="h-16 bg-white border-b flex items-center justify-between px-6 print:hidden">
                <div className="flex items-center space-x-4">
                    <button
                        className="text-gray-600 hover:text-gray-900"
                        onClick={() => navigate(`/documents/${documentId}`)}
                    >
                        <ChevronLeft className="w-6 h-6"/>
                    </button>
                    <div>
                        <h1 className="font-medium">{document.name}</h1>
                        <p className="text-sm text-gray-500">{document.project?.name}</p>
                    </div>
                </div>

                <div className="flex items-center space-x-3">
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button>
                                <Download className="w-4 h-4 mr-2"/>
                                Export
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent>
                            <DropdownMenuItem onClick={() => handleExport('pdf')}>
                                Export as PDF
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => handleExport('docx')}>
                                Export as DOCX
                            </DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                </div>
            </div>

            {/* Pages Section */}
            <div className="flex-1 flex print:block">
                {/* Pages Navigation */}
                <div className="w-64 border-r bg-white p-4 print:hidden">
                    <div className="mb-4">
                        <h2 className="text-sm font-medium text-gray-500">Pages</h2>
                    </div>

                    <div className="space-y-2">
                        {document.pages.map((page) => (
                            <button
                                key={page.id}
                                className={cn(
                                    "w-full flex items-center p-2 hover:bg-gray-50 rounded text-left text-sm",
                                    selectedPage === page.id && "bg-blue-50"
                                )}
                                onClick={() => setSelectedPage(page.id)}
                            >
                                <div className="relative w-12 h-16 bg-gray-100 rounded mr-3 flex-shrink-0 group">
                                    <img
                                        src={getStorageUrl(page.image_path)}
                                        alt={`Page ${page.page_number} thumbnail`}
                                        className="w-full h-full object-cover rounded"
                                    />
                                    <Button
                                        variant="secondary"
                                        size="sm"
                                        className="absolute bottom-0 right-0 opacity-0 group-hover:opacity-100 transition-opacity"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setPreviewImage(getStorageUrl(page.image_path));
                                        }}
                                    >
                                        <ZoomIn className="w-3 h-3"/>
                                    </Button>
                                </div>
                                <div>
                                    <div className="font-medium">Page {page.page_number}</div>
                                    <div className="text-xs text-gray-500">
                                        Writer: {page.writer?.name || 'Unassigned'}
                                    </div>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Preview Content */}
                <div className="flex-1 p-8 overflow-auto">
                    <DocumentContent document={document}/>
                </div>
            </div>

            {/* Share Dialog */}
            <ShareDialog
                open={showShareDialog}
                onOpenChange={setShowShareDialog}
                documentId={documentId}
            />

            {/* Image Preview Modal */}
            {previewImage && (
                <ImagePreviewModal
                    open={true}
                    onOpenChange={() => setPreviewImage(null)}
                    imageSrc={previewImage}
                />
            )}
        </div>
    );
}

// Document Content Component
interface DocumentContentProps {
    document: DocumentWithRelations;
}

function DocumentContent({document}: DocumentContentProps) {
    return (
        <div className="max-w-4xl mx-auto bg-white shadow-sm rounded-lg">
            <div className="p-8 prose max-w-none">
                <h1>{document?.name}</h1>

                {document?.pages?.map((page) => (
                    <div key={page.id} className="mb-8 page-break-after-always">
                        <h2 className="text-gray-500 print:text-black">
                            Page {page.page_number}
                        </h2>
                        <div
                            className="mt-4"
                            dangerouslySetInnerHTML={{
                                __html: page.formatted_text || page.extracted_text || ''
                            }}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
}

// Share Dialog Component
interface ShareDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    documentId?: string;
}

function ShareDialog({open, onOpenChange, documentId}: ShareDialogProps) {
    const [shareLink, setShareLink] = useState('');
    const [copied, setCopied] = useState(false);

    const generateShareLink = () => {
        const link = `${window.location.origin}/shared/documents/${documentId}`;
        setShareLink(link);
    };

    const copyToClipboard = async () => {
        try {
            await navigator.clipboard.writeText(shareLink);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <div className="space-y-4">
                    <div className="flex justify-between items-center">
                        <h2 className="text-lg font-medium">Share Document</h2>
                        <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
                            <X className="w-4 h-4"/>
                        </Button>
                    </div>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Share Link
                            </label>
                            <div className="flex space-x-2">
                                <input
                                    type="text"
                                    value={shareLink}
                                    readOnly
                                    className="flex-1 border rounded-lg p-2 text-sm bg-gray-50"
                                />
                                <Button onClick={copyToClipboard}>
                                    {copied ? 'Copied!' : 'Copy'}
                                </Button>
                            </div>
                        </div>

                        {!shareLink && (
                            <Button
                                onClick={generateShareLink}
                                className="w-full"
                            >
                                Generate Share Link
                            </Button>
                        )}
                    </div>

                    <div className="mt-4">
                        <h3 className="text-sm font-medium text-gray-700 mb-2">
                            Share Settings
                        </h3>
                        <div className="space-y-2">
                            <label className="flex items-center">
                                <input type="checkbox" className="rounded border-gray-300 mr-2"/>
                                <span className="text-sm">Allow comments</span>
                            </label>
                            <label className="flex items-center">
                                <input type="checkbox" className="rounded border-gray-300 mr-2"/>
                                <span className="text-sm">Allow downloads</span>
                            </label>
                            <label className="flex items-center">
                                <input type="checkbox" className="rounded border-gray-300 mr-2"/>
                                <span className="text-sm">Set expiration date</span>
                            </label>
                        </div>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}