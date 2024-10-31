import {useNavigate, useParams} from 'react-router-dom';
import {useMutation, useQuery, useQueryClient} from '@tanstack/react-query';
import {ChevronLeft, Loader2, Pencil, RefreshCw, Upload} from 'lucide-react';
import {Button} from '@/components/ui/button';
import {Card, CardContent} from '@/components/ui/card';
import {type TrainingSample, writerApi} from '@/services/api';
import {useEffect, useState} from 'react';
import {AddSampleDialog} from '@/components/dialogs/AddSampleDialog';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle
} from "@/components/ui/alert-dialog";
import WriterStatsDialog from '@/components/dialogs/WriterStatsDialog';
import ImagePreviewModal from '@/components/ImagePreviewModal';
import {EditTrainingSampleDialog} from '@/components/dialogs/EditTrainingSampleDialog';
import TrainingSampleCard from "@/components/TrainingSampleCard.tsx";
import {toast} from "sonner";

export default function WriterDetail() {
    const {writerId} = useParams();
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const [showAddSample, setShowAddSample] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
    const [showStats, setShowStats] = useState(false);
    const [selectedSample, setSelectedSample] = useState<number | null>(null);
    const [previewImage, setPreviewImage] = useState<string | null>(null);
    const [showEditSample, setShowEditSample] = useState<number | null>(null);
    const [sampleToEdit, setSampleToEdit] = useState<TrainingSample | null>(null);
    const [isEditingName, setIsEditingName] = useState(false);
    const [newSamples, setNewSamples] = useState<Set<number>>(new Set());
    const [samplesNeedingReview, setSamplesNeedingReview] = useState<Set<number>>(new Set());
    const [isTraining, setIsTraining] = useState(false);

    const updateSample = useMutation({
        mutationFn: async ({id, text}: { id: number, text: string }) => {
            return writerApi.updateTrainingSample(id, text);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['training-samples', writerId]});
            setShowEditSample(null);
            setSampleToEdit(null);
        },
    });

    // Fetch writer data
    const {data: writer} = useQuery({
        queryKey: ['writer', writerId],
        queryFn: async () => {
            const response = await writerApi.get(Number(writerId));
            return response.data;
        },
    });

    const [newName, setNewName] = useState(writer?.name || '');

    const updateWriter = useMutation({
        mutationFn: (data: { name: string }) => writerApi.update(Number(writerId), data),
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['writer', writerId]});
            queryClient.invalidateQueries({queryKey: ['writers']});
            setIsEditingName(false);
        },
    });

    // Fetch training samples
    const {data: trainingSamples} = useQuery({
        queryKey: ['training-samples', writerId],
        queryFn: async () => {
            const response = await writerApi.getTrainingSamples(Number(writerId));
            return response.data as TrainingSample[];
        },
    });

    // Delete sample mutation
    const deleteSample = useMutation({
        mutationFn: (sampleId: number) =>
            writerApi.deleteTrainingSample(sampleId),
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['training-samples', writerId]});
            setSelectedSample(null);
        },
    });

    // Start training mutation
    const startTraining = useMutation({
        mutationFn: async () => {
            if (!writerId) throw new Error('No writer ID');
            return writerApi.startTraining(Number(writerId));
        },
        onSuccess: () => {
            setIsTraining(true);
            queryClient.invalidateQueries({queryKey: ['writer', writerId]});
            startPolling();
        },
    });

    // Update the API response handling in handleAddSample
    const handleAddSample = async (file: File) => {
        if (!writerId) return;
        try {
            const response = await writerApi.newTrainingSample(Number(writerId), file);

            // If it's a multiline sample (array response)
            if (Array.isArray(response.data)) {
                const newIds = response.data
                    .filter(sample => sample.needs_review)
                    .map(sample => sample.id);
                setSamplesNeedingReview(prev => new Set([...prev, ...newIds]));
                setNewSamples(prev => new Set([...prev, ...newIds]));
            }
            // Single sample case
            else if (response.data.needs_review) {
                setSamplesNeedingReview(prev => new Set([...prev, response.data.id]));
                setNewSamples(prev => new Set([...prev, response.data.id]));
            }

            // Don't immediately invalidate the query - let the UI update first
            setTimeout(() => {
                queryClient.invalidateQueries({queryKey: ['training-samples', writerId]});
            }, 100);

            toast.success('Training samples processed successfully');
        } catch (error) {
            console.error('Failed to process training sample:', error);
            toast.error('Failed to process training sample');
        }
    };

    const pollTrainingStatus = async () => {
        if (!writerId) return;
        try {
            const response = await writerApi.get(Number(writerId));
            // Assuming the API returns some indication that training is complete
            // You might need to adjust this based on your actual API response
            if (response.data.status !== 'training') {
                setIsTraining(false);
                queryClient.invalidateQueries({queryKey: ['writer', writerId]});
                return true; // Training complete
            }
            return false; // Still training
        } catch (error) {
            console.error('Error polling training status:', error);
            setIsTraining(false);
            return true; // Stop polling on error
        }
    };

    const startPolling = () => {
        const poll = async () => {
            const isComplete = await pollTrainingStatus();
            if (!isComplete) {
                setTimeout(poll, 5000); // Poll every 5 seconds
            }
        };
        poll();
    };

    // Clean up polling on unmount
    useEffect(() => {
        return () => {
            setIsTraining(false);
        };
    }, []);

    // Add this effect after your existing useEffect hooks
    useEffect(() => {
        if (trainingSamples) {
            setSamplesNeedingReview(prev => {
                const newSet = new Set(prev); // Keep existing flags
                trainingSamples.forEach(sample => {
                    if (sample.needs_review) {
                        newSet.add(sample.id);
                    }
                });
                return newSet;
            });
        }
    }, [trainingSamples]);

    useEffect(() => {
        const timeouts: NodeJS.Timeout[] = [];

        newSamples.forEach(sampleId => {
            const timeout = setTimeout(() => {
                setNewSamples(prev => {
                    const next = new Set(prev);
                    next.delete(sampleId);
                    return next;
                });
            }, 5 * 60 * 1000); // 5 minutes
            timeouts.push(timeout);
        });

        return () => timeouts.forEach(clearTimeout);
    }, [newSamples]);

    if (!writer) {
        return <div className="p-8">Loading...</div>;
    }

    return (
        <div className="p-8">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                    <button
                        className="flex items-center text-sm text-gray-500 mb-4 hover:text-gray-700"
                        onClick={() => navigate('/writers')}
                    >
                        <ChevronLeft className="w-4 h-4 mr-1"/>
                        Back to Writers
                    </button>

                    <div className="flex justify-between items-center">
                        <div>
                            <div className="flex items-center gap-2">
                                {isEditingName ? (
                                    <div className="flex items-center gap-2">
                                        <input
                                            type="text"
                                            value={newName}
                                            onChange={(e) => setNewName(e.target.value)}
                                            className="text-2xl font-bold bg-white border rounded px-2 py-1"
                                            autoFocus
                                        />
                                        <Button
                                            size="sm"
                                            onClick={() => {
                                                if (newName.trim()) {
                                                    updateWriter.mutate({name: newName.trim()});
                                                }
                                            }}
                                            disabled={updateWriter.isPending}
                                        >
                                            Save
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={() => {
                                                setIsEditingName(false);
                                                setNewName(writer.name);
                                            }}
                                        >
                                            Cancel
                                        </Button>
                                    </div>
                                ) : (
                                    <>
                                        <h1 className="text-2xl font-bold">{writer.name}</h1>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => {
                                                setIsEditingName(true);
                                                setNewName(writer.name);
                                            }}
                                        >
                                            <Pencil className="w-4 h-4"/>
                                        </Button>
                                    </>
                                )}
                            </div>
                            <p className="text-gray-500">
                                {writer.last_trained
                                    ? `Last trained ${new Date(writer.last_trained).toLocaleDateString()}`
                                    : 'Not trained yet'}
                            </p>
                        </div>

                        <div className="flex space-x-3">
                            <Button onClick={() => setShowAddSample(true)}>
                                <Upload className="w-4 h-4 mr-2"/>
                                Add Sample
                            </Button>

                            {trainingSamples && trainingSamples.length > 0 && (
                                <Button
                                    variant="outline"
                                    onClick={() => startTraining.mutate()}
                                    disabled={startTraining.isPending || isTraining}
                                >
                                    {startTraining.isPending || isTraining ? (
                                        <>
                                            <Loader2 className="w-4 h-4 mr-2 animate-spin"/>
                                            Training in progress...
                                        </>
                                    ) : (
                                        <>
                                            <RefreshCw className="w-4 h-4 mr-2"/>
                                            Start Training
                                        </>
                                    )}
                                </Button>
                            )}
                        </div>
                    </div>
                </div>

                {/* Training Samples Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {trainingSamples?.map((sample) => (
                        <TrainingSampleCard
                            key={sample.id}
                            sample={sample}
                            isNew={newSamples.has(sample.id)}
                            needsReview={samplesNeedingReview.has(sample.id)}
                            onEdit={() => {
                                setShowEditSample(sample.id);
                                setSampleToEdit(sample);
                                // Remove from review when editing
                                setSamplesNeedingReview(prev => {
                                    const next = new Set(prev);
                                    next.delete(sample.id);
                                    return next;
                                });
                            }}
                            onDelete={() => {
                                setSelectedSample(sample.id);
                                setShowDeleteConfirm(true);
                            }}
                        />
                    ))}

                    {/* Add Sample Card */}
                    <Card
                        className="border-2 border-dashed cursor-pointer hover:border-blue-400 hover:bg-blue-50/50 transition-colors"
                        onClick={() => setShowAddSample(true)}
                    >
                        <CardContent className="p-8 flex flex-col items-center justify-center text-center">
                            <Upload className="w-6 h-6 text-gray-400 mb-2"/>
                            <p className="text-sm text-gray-500">
                                Add new training sample
                            </p>
                        </CardContent>
                    </Card>
                </div>
            </div>

            {/* Dialogs */}
            <AddSampleDialog
                open={showAddSample}
                onOpenChange={setShowAddSample}
                onSubmit={handleAddSample}
            />

            <AlertDialog
                open={showDeleteConfirm}
                onOpenChange={setShowDeleteConfirm}
            >
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Training Sample</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete this training sample?
                            This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={() => {
                                if (selectedSample) {
                                    deleteSample.mutate(selectedSample);
                                }
                                setShowDeleteConfirm(false);
                            }}
                            className="bg-red-600 hover:bg-red-700"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {showStats && (
                <WriterStatsDialog
                    writerId={Number(writerId)}
                    open={true}
                    onOpenChange={setShowStats}
                />
            )}

            {previewImage && (
                <ImagePreviewModal
                    open={true}
                    onOpenChange={() => setPreviewImage(null)}
                    imageSrc={previewImage}
                />
            )}
            <EditTrainingSampleDialog
                open={showEditSample !== null}
                onOpenChange={(open) => {
                    if (!open) {
                        setShowEditSample(null);
                        setSampleToEdit(null);
                    }
                }}
                sample={sampleToEdit}
                onSubmit={(id, text) => updateSample.mutate({id, text})}
            />
        </div>
    );
}