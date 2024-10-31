import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { Plus, FileText, Trash2 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { projectApi } from '@/services/api';
import NewProjectDialog from '@/components/dialogs/NewProjectDialog';
import { useState } from 'react';
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

export default function ProjectList() {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [showNewProject, setShowNewProject] = useState(false);
    const [projectToDelete, setProjectToDelete] = useState<number | null>(null);

    const { data: projects, isLoading } = useQuery({
        queryKey: ['projects'],
        queryFn: async () => {
            const response = await projectApi.list();
            return response.data;
        },
    });

    const deleteProject = useMutation({
        mutationFn: projectApi.delete,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['projects'] });
            setProjectToDelete(null);
        },
    });

    if (isLoading) {
        return <div className="p-8">Loading projects...</div>;
    }

    return (
        <div className="p-8">
            <div className="max-w-5xl mx-auto">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h1 className="text-2xl font-bold">Projects</h1>
                        <p className="text-gray-500">Manage your document projects</p>
                    </div>

                    <Button onClick={() => setShowNewProject(true)}>
                        <Plus className="w-4 h-4 mr-2" />
                        New Project
                    </Button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {projects?.map((project) => (
                        <Card
                            key={project.id}
                            className="hover:shadow-md transition-shadow"
                        >
                            <CardContent className="p-6">
                                <div
                                    className="flex items-center space-x-3 cursor-pointer"
                                    onClick={() => navigate(`/projects/${project.id}`)}
                                >
                                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                                        <FileText className="w-5 h-5 text-blue-600" />
                                    </div>
                                    <div>
                                        <h3 className="font-medium">{project.name}</h3>
                                        <p className="text-sm text-gray-500">
                                            Created {new Date(project.created_at).toLocaleDateString()}
                                        </p>
                                    </div>
                                </div>

                                {project.description && (
                                    <p className="mt-3 text-sm text-gray-600">{project.description}</p>
                                )}

                                <div className="mt-4 flex justify-end">
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setProjectToDelete(project.id);
                                        }}
                                    >
                                        <Trash2 className="w-4 h-4 mr-2" />
                                        Delete
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </div>

            <NewProjectDialog
                open={showNewProject}
                onOpenChange={setShowNewProject}
            />

            <AlertDialog open={projectToDelete !== null} onOpenChange={() => setProjectToDelete(null)}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Project</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete this project? This action cannot be undone.
                            All documents and pages within this project will be permanently deleted.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={() => projectToDelete && deleteProject.mutate(projectToDelete)}
                            className="bg-red-600 hover:bg-red-700 text-white"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
}