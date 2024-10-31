import {useQuery} from '@tanstack/react-query';
import {Link, useLocation, useNavigate} from 'react-router-dom';
import {ChevronDown, ChevronRight, ChevronUp, FileText, Plus, User} from 'lucide-react';
import {cn} from '@/lib/utils';
import {projectApi, writerApi} from '@/services/api';
import {useState} from 'react';
import {ScrollArea} from '@/components/ui/scroll-area';
import NewProjectDialog from '@/components/dialogs/NewProjectDialog';
import NewWriterDialog from '@/components/dialogs/NewWriterDialog';
import {Button} from '@/components/ui/button';
import {Separator} from '@/components/ui/separator';

export default function Sidebar() {
    const location = useLocation();
    const navigate = useNavigate();
    const [showNewProject, setShowNewProject] = useState(false);
    const [showNewWriter, setShowNewWriter] = useState(false);
    const [isWritersExpanded, setIsWritersExpanded] = useState(true);
    const [isProjectsExpanded, setIsProjectsExpanded] = useState(true);

    // Fetch projects and writers
    const {data: projects, isError: isProjectsError} = useQuery({
        queryKey: ['projects'],
        queryFn: async () => {
            const response = await projectApi.list();
            return response.data;
        },
    });

    const {data: writers, isError: isWritersError} = useQuery({
        queryKey: ['writers'],
        queryFn: async () => {
            const response = await writerApi.list();
            return response.data;
        },
    });

    const currentProjectId = location.pathname.match(/\/projects\/(\d+)/)?.[1];
    const currentWriterId = location.pathname.match(/\/writers\/(\d+)/)?.[1];

    return (
        <aside className="w-64 bg-white border-r border-gray-200 flex flex-col h-screen">
            {/* Fixed Header */}
            <div className="flex-none p-4 border-b">
                <Link to="/" className="text-xl font-bold text-gray-900 block mb-4">
                    HandScript
                </Link>
            </div>

            {/* Scrollable Content */}
            <ScrollArea className="flex-1">
                <div className="px-2 py-4">
                    {/* Projects Section */}
                    <div className="mb-6">
                        <div className="px-2 mb-2 flex items-center justify-between">
                            <button
                                className="flex items-center text-sm font-semibold text-gray-500 hover:text-gray-900"
                                onClick={() => setIsProjectsExpanded(!isProjectsExpanded)}
                            >
                                {isProjectsExpanded ? (
                                    <ChevronDown className="w-4 h-4 mr-1"/>
                                ) : (
                                    <ChevronUp className="w-4 h-4 mr-1"/>
                                )}
                                Projects
                            </button>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowNewProject(true)}
                            >
                                <Plus className="w-4 h-4"/>
                            </Button>
                        </div>
                        {isProjectsExpanded && (
                            <div className="space-y-0.5">
                                {/* All Projects Link */}
                                <button
                                    className={cn(
                                        "w-full text-left px-2 py-2 flex items-center rounded-md text-sm",
                                        location.pathname === '/projects'
                                            ? "bg-blue-50 text-blue-600"
                                            : "hover:bg-gray-50 text-gray-700"
                                    )}
                                    onClick={() => navigate('/projects')}
                                >
                                    <FileText className="w-4 h-4 mr-2"/>
                                    <span>All Projects</span>
                                </button>

                                {/* Individual Projects */}
                                {isProjectsError ? (
                                    <div className="px-2 py-2 text-sm text-red-600">
                                        Failed to load projects
                                    </div>
                                ) : (
                                    projects?.map((project) => (
                                        <button
                                            key={project.id}
                                            className={cn(
                                                "w-full text-left px-2 py-2 flex items-center rounded-md text-sm",
                                                project.id === Number(currentProjectId)
                                                    ? "bg-blue-50 text-blue-600"
                                                    : "hover:bg-gray-50 text-gray-700"
                                            )}
                                            onClick={() => navigate(`/projects/${project.id}`)}
                                        >
                                            <FileText className={cn(
                                                "w-4 h-4 mr-2",
                                                project.id === Number(currentProjectId)
                                                    ? "text-blue-600"
                                                    : "text-gray-400"
                                            )}/>
                                            <span className="truncate flex-1">{project.name}</span>
                                            {project.id === Number(currentProjectId) && (
                                                <ChevronRight className="w-4 h-4 ml-2 flex-none"/>
                                            )}
                                        </button>
                                    ))
                                )}
                            </div>
                        )}
                    </div>

                    <Separator className="my-4"/>

                    {/* Writers Section */}
                    <div>
                        <div className="px-2 mb-2 flex items-center justify-between">
                            <button
                                className="flex items-center text-sm font-semibold text-gray-500 hover:text-gray-900"
                                onClick={() => setIsWritersExpanded(!isWritersExpanded)}
                            >
                                {isWritersExpanded ? (
                                    <ChevronDown className="w-4 h-4 mr-1"/>
                                ) : (
                                    <ChevronUp className="w-4 h-4 mr-1"/>
                                )}
                                Writers
                            </button>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowNewWriter(true)}
                            >
                                <Plus className="w-4 h-4"/>
                            </Button>
                        </div>

                        {isWritersExpanded && (
                            <div className="space-y-0.5">
                                {/* Writers Management Link */}
                                <button
                                    className={cn(
                                        "w-full text-left px-2 py-2 flex items-center rounded-md text-sm",
                                        location.pathname === '/writers'
                                            ? "bg-blue-50 text-blue-600"
                                            : "hover:bg-gray-50 text-gray-700"
                                    )}
                                    onClick={() => navigate('/writers')}
                                >
                                    <User className="w-4 h-4 mr-2"/>
                                    <span>All Writers</span>
                                </button>

                                {/* Individual Writers */}
                                {isWritersError ? (
                                    <div className="px-2 py-2 text-sm text-red-600">
                                        Failed to load writers
                                    </div>
                                ) : (
                                    writers?.map((writer) => (
                                        <button
                                            key={writer.id}
                                            className={cn(
                                                "w-full text-left px-2 py-2 flex items-center rounded-md text-sm",
                                                writer.id === Number(currentWriterId)
                                                    ? "bg-blue-50 text-blue-600"
                                                    : "hover:bg-gray-50 text-gray-700"
                                            )}
                                            onClick={() => navigate(`/writers/${writer.id}`)}
                                        >
                                            <div className="w-4 h-4 mr-2 flex items-center justify-center">
                                                <div
                                                    className={cn(
                                                        "w-2 h-2 rounded-full",
                                                        writer.status === 'ready' ? "bg-green-400" :
                                                            writer.status === 'training' ? "bg-blue-400" :
                                                                "bg-yellow-400"
                                                    )}
                                                />
                                            </div>
                                            <span className="truncate flex-1">{writer.name}</span>
                                            {writer.id === Number(currentWriterId) && (
                                                <ChevronRight className="w-4 h-4 ml-2 flex-none"/>
                                            )}
                                        </button>
                                    ))
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </ScrollArea>

            {/* Dialogs */}
            <NewProjectDialog
                open={showNewProject}
                onOpenChange={setShowNewProject}
            />
            <NewWriterDialog
                open={showNewWriter}
                onOpenChange={setShowNewWriter}
            />
        </aside>
    );
}