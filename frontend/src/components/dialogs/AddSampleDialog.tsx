import {Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle} from '@/components/ui/dialog';
import {Button} from '@/components/ui/button';
import {useEffect, useState} from 'react';
import {Upload, X} from 'lucide-react';
import {Form} from "@/components/ui/form";
import {useForm} from "react-hook-form";
import {toast} from "sonner";

interface AddSampleDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onSubmit: (file: File, text: string) => void;
}

export function AddSampleDialog({
                                    open,
                                    onOpenChange,
                                    onSubmit
                                }: AddSampleDialogProps) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);

    const form = useForm();

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
            setSelectedFile(file);
        }
    };

    const handleSubmit = async () => {
        if (!selectedFile) return;

        setIsProcessing(true);
        try {
            // Use the new process endpoint that handles segmentation
            await onSubmit(selectedFile, ''); // Text parameter is now unused
            toast.success('Training sample processed and added successfully');
            onOpenChange(false);
        } catch (error) {
            toast.error('Error processing training sample');
            console.error('Error:', error);
        } finally {
            setIsProcessing(false);
            setSelectedFile(null);
            setPreviewUrl(null);
        }
    };

    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Add Training Sample</DialogTitle>
                </DialogHeader>

                <Form {...form}>
                    <form className="space-y-4">
                        <div className="border-2 border-dashed rounded-lg p-4">
                            {selectedFile ? (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <span className="text-sm text-gray-600">
                                            {selectedFile.name}
                                        </span>
                                        <Button
                                            type="button"
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => {
                                                if (previewUrl) URL.revokeObjectURL(previewUrl);
                                                setPreviewUrl(null);
                                                setSelectedFile(null);
                                            }}
                                        >
                                            <X className="w-4 h-4"/>
                                        </Button>
                                    </div>
                                    <div className="relative w-full h-48 bg-gray-100 rounded overflow-hidden">
                                        {previewUrl && (
                                            <img
                                                src={previewUrl}
                                                alt="Preview"
                                                className="w-full h-full object-contain"
                                            />
                                        )}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center">
                                    <Upload className="w-6 h-6 text-gray-400 mx-auto mb-2"/>
                                    <input
                                        type="file"
                                        onChange={handleFileSelect}
                                        accept="image/*"
                                        className="hidden"
                                        id="sample-image"
                                    />
                                    <label
                                        htmlFor="sample-image"
                                        className="text-sm text-blue-600 hover:text-blue-700 cursor-pointer"
                                    >
                                        Select image file
                                    </label>
                                    <p className="mt-2 text-xs text-gray-500">
                                        Image will be processed and split into line samples automatically
                                    </p>
                                </div>
                            )}
                        </div>

                        <DialogFooter>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => onOpenChange(false)}
                                disabled={isProcessing}
                            >
                                Cancel
                            </Button>
                            <Button
                                type="button"
                                onClick={handleSubmit}
                                disabled={!selectedFile || isProcessing}
                            >
                                {isProcessing ? (
                                    <>
                                        <div
                                            className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"/>
                                        Processing...
                                    </>
                                ) : (
                                    'Add Samples'
                                )}
                            </Button>
                        </DialogFooter>
                    </form>
                </Form>
            </DialogContent>
        </Dialog>
    );
}