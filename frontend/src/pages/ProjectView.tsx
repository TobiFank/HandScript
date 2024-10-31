// src/pages/ProjectView.tsx
import {useNavigate, useParams} from 'react-router-dom';
import {useMutation, useQuery, useQueryClient} from '@tanstack/react-query';
import {FileText, Plus, Trash2, Upload} from 'lucide-react';
import {Card, CardContent} from '@/components/ui/card';
import {Button} from '@/components/ui/button';
import {documentApi, projectApi} from '@/services/api';
import NewDocumentDialog from '@/components/dialogs/NewDocumentDialog';
import {useState} from 'react';
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

export default function ProjectView() {
    const {projectId} = useParams();
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [showNewDocument, setShowNewDocument] = useState(false);
    const [documentToDelete, setDocumentToDelete] = useState<number | null>(null);

    const {data: project} = useQuery({
        queryKey: ['project', projectId],
        queryFn: async () => {
            const response = await projectApi.get(Number(projectId));
            return response.data;
        },
    });

    const {data: documents, isLoading} = useQuery({
        queryKey: ['documents', projectId],
        queryFn: async () => {
            const response = await documentApi.get(Number(projectId));
            return response.data;
        },
        enabled: !!projectId,
    });

    const createDocument = useMutation({
        mutationFn: async (data: { name: string; description?: string }) => {
            if (!projectId) throw new Error('No project ID');
            const response = await documentApi.create(Number(projectId), data);
            return response.data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['documents', projectId]});
            setShowNewDocument(false);
        },
    });

    const deleteDocument = useMutation({
        mutationFn: (id: number) => documentApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['documents', projectId]});
            setDocumentToDelete(null);
        },
    });

    if (isLoading) {
        return <div className="p-8">Loading documents...</div>;
    }

    return (
        <div className="p-8">
            <div className="max-w-5xl mx-auto">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h1 className="text-2xl font-bold">{project?.name}</h1>
                        <p className="text-gray-500">
                            {project?.description || 'Manage documents within this project'}
                        </p>
                    </div>

                    <Button onClick={() => setShowNewDocument(true)}>
                        <Plus className="w-4 h-4 mr-2"/>
                        New Document
                    </Button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {documents?.map((document) => (
                        <Card
                            key={document.id}
                            className="hover:shadow-md transition-shadow cursor-pointer"
                            onClick={() => navigate(`/documents/${document.id}`)}
                        >
                            <CardContent className="p-6">
                                <div className="flex items-center space-x-3">
                                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                                        <FileText className="w-5 h-5 text-blue-600"/>
                                    </div>
                                    <div>
                                        <h3 className="font-medium">{document.name}</h3>
                                        <p className="text-sm text-gray-500">
                                            Created {new Date(document.created_at).toLocaleDateString()}
                                        </p>
                                    </div>
                                </div>
                                {document.description && (
                                    <p className="mt-3 text-sm text-gray-600">{document.description}</p>
                                )}
                                <div className="mt-4 flex justify-end space-x-2">
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setDocumentToDelete(document.id);
                                        }}
                                    >
                                        <Trash2 className="w-4 h-4 mr-2"/>
                                        Delete
                                    </Button>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            navigate(`/documents/${document.id}`);
                                        }}
                                    >
                                        <Upload className="w-4 h-4 mr-2"/>
                                        Add Pages
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>

            <NewDocumentDialog
                open={showNewDocument}
                onOpenChange={setShowNewDocument}
                onSubmit={(data) => createDocument.mutate(data)}
            />
            <AlertDialog open={documentToDelete !== null} onOpenChange={() => setDocumentToDelete(null)}>
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
                            onClick={() => documentToDelete && deleteDocument.mutate(documentToDelete)}
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