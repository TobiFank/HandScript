import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useState } from 'react';
import { Upload, X, Info } from 'lucide-react';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
} from "@/components/ui/form";
import { useForm } from "react-hook-form";

interface TrainingSampleDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onSubmit: (files: File[], texts: string[]) => void;
    language: 'english' | 'german';
    trainingType: 'quick' | 'full';
}

// Define minimum required training samples
const REQUIRED_SAMPLES = 3; // Adjust this number based on your requirements

export default function TrainingSamplesDialog({
                                                  open,
                                                  onOpenChange,
                                                  onSubmit,
                                              }: TrainingSampleDialogProps) {
    const [uploadedFiles, setUploadedFiles] = useState<(File | null)[]>(
        new Array(REQUIRED_SAMPLES).fill(null)
    );

    const form = useForm({
        defaultValues: {
            samples: new Array(REQUIRED_SAMPLES).fill(''),
        },
    });

    const handleFileSelect = (index: number, file: File | null) => {
        setUploadedFiles(prev => {
            const newFiles = [...prev];
            newFiles[index] = file;
            return newFiles;
        });
    };

    const handleSubmit = (data: { samples: string[] }) => {
        const validFiles = uploadedFiles.filter((f): f is File => f !== null);
        const validTexts = data.samples.filter(text => text.trim());
        onSubmit(validFiles, validTexts);
    };

    const completionPercentage =
        (uploadedFiles.filter(f => f !== null).length / uploadedFiles.length) * 100;

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[700px]">
                <DialogHeader>
                    <DialogTitle>Training Samples Setup</DialogTitle>
                </DialogHeader>

                <div className="space-y-6">
                    <Alert>
                        <Info className="h-4 w-4" />
                        <AlertDescription>
                            Please write each sentence on a separate piece of paper, using your natural handwriting.
                            Scan or photograph each sample clearly and upload them below.
                        </AlertDescription>
                    </Alert>

                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span>Upload Progress</span>
                            <span>{Math.round(completionPercentage)}%</span>
                        </div>
                        <Progress value={completionPercentage} />
                    </div>

                    <Form {...form}>
                        <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
                            {Array.from({ length: REQUIRED_SAMPLES }).map((_, index) => (
                                <div key={index} className="border rounded-lg p-4 space-y-3">
                                    <FormField
                                        control={form.control}
                                        name={`samples.${index}`}
                                        render={({ field }) => (
                                            <FormItem>
                                                <FormLabel>Sample Text {index + 1}</FormLabel>
                                                <FormControl>
                                                    <div className="space-y-3">
                                                        <textarea
                                                            {...field}
                                                            className="w-full min-h-[60px] rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                                            placeholder="Enter the text for this sample..."
                                                        />

                                                        <div className="border-2 border-dashed rounded-lg p-4">
                                                            {uploadedFiles[index] ? (
                                                                <div className="flex items-center justify-between">
                                                                    <span className="text-sm text-gray-600">
                                                                        {uploadedFiles[index]?.name}
                                                                    </span>
                                                                    <Button
                                                                        type="button"
                                                                        variant="ghost"
                                                                        size="sm"
                                                                        onClick={() => handleFileSelect(index, null)}
                                                                    >
                                                                        <X className="w-4 h-4" />
                                                                    </Button>
                                                                </div>
                                                            ) : (
                                                                <div className="text-center">
                                                                    <Upload className="w-6 h-6 text-gray-400 mx-auto mb-2" />
                                                                    <input
                                                                        type="file"
                                                                        id={`file-${index}`}
                                                                        className="hidden"
                                                                        onChange={(e) => handleFileSelect(
                                                                            index,
                                                                            e.target.files?.[0] || null
                                                                        )}
                                                                        accept="image/*"
                                                                    />
                                                                    <label
                                                                        htmlFor={`file-${index}`}
                                                                        className="text-sm text-blue-600 hover:text-blue-700 cursor-pointer"
                                                                    >
                                                                        Upload Image
                                                                    </label>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                </FormControl>
                                            </FormItem>
                                        )}
                                    />
                                </div>
                            ))}

                            <DialogFooter>
                                <Button
                                    type="button"
                                    variant="outline"
                                    onClick={() => onOpenChange(false)}
                                >
                                    Cancel
                                </Button>
                                <Button
                                    type="submit"
                                    disabled={uploadedFiles.some(f => f === null) ||
                                        form.getValues().samples.some(text => !text.trim())}
                                >
                                    Start Training
                                </Button>
                            </DialogFooter>
                        </form>
                    </Form>
                </div>
            </DialogContent>
        </Dialog>
    );
}

