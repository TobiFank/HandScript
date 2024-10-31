// src/pages/WriterManagement.tsx
import {useMutation, useQuery, useQueryClient} from '@tanstack/react-query';
import {AlertTriangle, BarChart, Check, Plus, Trash2, User,} from 'lucide-react';
import {Card, CardContent} from '@/components/ui/card';
import {Button} from '@/components/ui/button';
import {type Writer, writerApi} from '@/services/api';
import {useState} from 'react';
import NewWriterDialog from '@/components/dialogs/NewWriterDialog';
import WriterStatsDialog from '@/components/dialogs/WriterStatsDialog.tsx';
import TrainingSamplesDialog from '@/components/dialogs/TrainingSamplesDialog';
import {Progress} from '@/components/ui/progress';
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
import {useNavigate} from "react-router-dom";

export default function WriterManagement() {
    const [showNewWriter, setShowNewWriter] = useState(false);
    const [showStats, setShowStats] = useState<number | null>(null);
    const [showTraining, setShowTraining] = useState<number | null>(null);
    const [writerToDelete, setWriterToDelete] = useState<number | null>(null);

    const queryClient = useQueryClient();

    const {data: writers} = useQuery({
        queryKey: ['writers'],
        queryFn: async () => {
            const response = await writerApi.list();
            return response.data as Writer[];
        },
    });

    const deleteWriter = useMutation({
        mutationFn: (id: number) => writerApi.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['writers']});
            setWriterToDelete(null);
        },
    });

    const trainWriter = useMutation({
        mutationFn: (data: { id: number; files: File[]; texts: string[] }) => {
            return writerApi.train(data.id, data.files, data.texts);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['writers']});
            setShowTraining(null);
        },
    });

    return (
        <div className="p-6 max-w-6xl mx-auto">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-2xl font-bold">Writers</h1>
                    <p className="text-gray-500">Manage writer profiles and handwriting models</p>
                </div>

                <Button onClick={() => setShowNewWriter(true)}>
                    <Plus className="w-4 h-4 mr-2"/>
                    Add Writer
                </Button>
            </div>

            {/* Writers Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {writers?.map((writer) => (
                    <WriterCard
                        key={writer.id}
                        writer={writer}
                        onDelete={() => setWriterToDelete(writer.id)}
                        onViewStats={() => setShowStats(writer.id)}
                        onTrain={() => setShowTraining(writer.id)}
                    />
                ))}
            </div>

            {/* Dialogs */}
            <NewWriterDialog
                open={showNewWriter}
                onOpenChange={setShowNewWriter}
            />

            {showStats !== null && (
                <WriterStatsDialog
                    writerId={showStats}
                    open={true}
                    onOpenChange={() => setShowStats(null)}
                />
            )}

            {showTraining !== null && (
                <TrainingSamplesDialog
                    open={true}
                    onOpenChange={() => setShowTraining(null)}
                    onSubmit={(files, texts) => {
                        trainWriter.mutate({
                            id: showTraining,
                            files,
                            texts
                        });
                    }}
                    language="english"
                    trainingType="quick"
                />
            )}

            <AlertDialog open={writerToDelete !== null} onOpenChange={() => setWriterToDelete(null)}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Writer Profile</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete this writer profile? This action cannot be undone.
                            All associated model data will be permanently removed.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={() => writerToDelete && deleteWriter.mutate(writerToDelete)}
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

// Writer Card Component
interface WriterCardProps {
    writer: Writer;
    onDelete: () => void;
    onViewStats: () => void;
    onTrain: () => void;
}

function WriterCard({writer, onDelete, onViewStats}: WriterCardProps) {
    const navigate = useNavigate();
    return (
        <Card>
            <CardContent
                className="p-6 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => navigate(`/writers/${writer.id}`)}
            >
                <div className="flex justify-between items-start">
                    <div className="flex items-center">
                        <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                            <User className="w-6 h-6 text-blue-600"/>
                        </div>
                        <div className="ml-4">
                            <h3 className="font-medium">{writer.name}</h3>
                            <p className="text-sm text-gray-500">
                                Added {new Date(writer.created_at).toLocaleDateString()}
                            </p>
                        </div>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full ${
                        writer.status === 'ready'
                            ? 'bg-green-100 text-green-700'
                            : writer.status === 'training'
                                ? 'bg-blue-100 text-blue-700'
                                : 'bg-yellow-100 text-yellow-700'
                    }`}>
                        {writer.status.charAt(0).toUpperCase() + writer.status.slice(1)}
                    </span>
                </div>

                <div className="mt-6 space-y-4">
                    {writer.status === 'training' && (
                        <div>
                            <div className="flex justify-between text-sm mb-2">
                                <span>Training Progress</span>
                                <span>75%</span>
                            </div>
                            <Progress value={75}/>
                        </div>
                    )}

                    <div className="grid grid-cols-2 gap-4">
                        <div className="text-sm">
                            <span className="text-gray-600">Accuracy</span>
                            <p className="font-medium">{writer.accuracy?.toFixed(1)}%</p>
                        </div>
                        <div className="text-sm">
                            <span className="text-gray-600">Pages Processed</span>
                            <p className="font-medium">{writer.pages_processed}</p>
                        </div>
                        {writer.last_trained && (
                            <div className="text-sm col-span-2">
                                <span className="text-gray-600">Last Trained</span>
                                <p className="font-medium">
                                    {writer.last_trained
                                        ? new Date(writer.last_trained).toLocaleDateString()
                                        : 'Never'}
                                </p>
                            </div>
                        )}
                    </div>

                    <div className="flex flex-wrap gap-2" onClick={e => e.stopPropagation()}>
                        {writer.status === 'ready' && (
                            <>
                                <Button variant="outline" size="sm" onClick={onViewStats}>
                                    <BarChart className="w-4 h-4 mr-2"/>
                                    View Stats
                                </Button>
                            </>
                        )}
                        {/* Delete button now always visible */}
                        <Button
                            variant="outline"
                            size="sm"
                            className="text-red-600 hover:text-red-700 hover:bg-red-50"
                            onClick={onDelete}
                        >
                            <Trash2 className="w-4 h-4 mr-2"/>
                            Delete
                        </Button>
                    </div>

                    {writer.status === 'training' && (
                        <div className="text-sm space-y-2">
                            <div className="flex items-center text-yellow-600">
                                <AlertTriangle className="w-4 h-4 mr-2"/>
                                Needed: Numbers (0-9)
                            </div>
                            <div className="flex items-center text-green-600">
                                <Check className="w-4 h-4 mr-2"/>
                                Complete: Capital letters
                            </div>
                            <div className="flex items-center text-green-600">
                                <Check className="w-4 h-4 mr-2"/>
                                Complete: Lowercase letters
                            </div>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}