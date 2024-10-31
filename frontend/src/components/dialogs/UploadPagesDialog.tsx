import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useQuery } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { writerApi, type Writer } from '@/services/api';
import { Upload, X } from "lucide-react";
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
} from "@/components/ui/form";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { useForm } from "react-hook-form";

interface UploadPagesDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    onUpload: (files: File[], writerId: number) => void;
    isUploading: boolean;
}

interface FormData {
    writerId: string;
}

export default function UploadPagesDialog({
                                              open,
                                              onOpenChange,
                                              onUpload,
                                              isUploading,
                                          }: UploadPagesDialogProps) {
    const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
    const [previewUrls, setPreviewUrls] = useState<string[]>([]);

    const form = useForm<FormData>({
        defaultValues: {
            writerId: '',
        }
    });

    const { data: writers } = useQuery({
        queryKey: ['writers'],
        queryFn: async () => {
            const response = await writerApi.list();
            return response.data as Writer[];
        },
    });


    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        const urls = files.map(file => URL.createObjectURL(file));
        setPreviewUrls(prevUrls => [...prevUrls, ...urls]);
        setSelectedFiles(prev => [...prev, ...files]);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const files = Array.from(e.dataTransfer.files);
        const urls = files.map(file => URL.createObjectURL(file));
        setPreviewUrls(prevUrls => [...prevUrls, ...urls]);
        setSelectedFiles(files);
    };

// Add cleanup
    useEffect(() => {
        return () => {
            previewUrls.forEach(url => URL.revokeObjectURL(url));
        };
    }, [previewUrls]);

    const handleSubmit = async (data: FormData) => {
        if (selectedFiles.length > 0 && data.writerId) {
            await onUpload(selectedFiles, Number(data.writerId));
        }
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent onPointerDownOutside={(e) => {
                if (isUploading) {
                    e.preventDefault();
                }
            }}>
                <DialogHeader>
                    <DialogTitle>Upload Pages</DialogTitle>
                </DialogHeader>

                <Form {...form}>
                    <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
                        <div
                            className="border-2 border-dashed border-gray-200 rounded-lg p-8 text-center"
                            onDrop={handleDrop}
                            onDragOver={(e) => e.preventDefault()}
                        >
                            {selectedFiles.length > 0 ? (
                                <div className="space-y-4">
                                    {selectedFiles.map((file, index) => (
                                        <div key={index} className="space-y-2">
                                            <div className="flex items-center justify-between">
                                                <span className="text-sm text-gray-600">{file.name}</span>
                                                <Button
                                                    type="button"
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={() => {
                                                        if (previewUrls[index]) URL.revokeObjectURL(previewUrls[index]);
                                                        const newFiles = [...selectedFiles];
                                                        newFiles.splice(index, 1);
                                                        setSelectedFiles(newFiles);
                                                        const newUrls = [...previewUrls];
                                                        newUrls.splice(index, 1);
                                                        setPreviewUrls(newUrls);
                                                    }}
                                                >
                                                    <X className="w-4 h-4" />
                                                </Button>
                                            </div>
                                            <div className="relative w-full h-48 bg-gray-100 rounded overflow-hidden">
                                                <img
                                                    src={previewUrls[index]}
                                                    alt={`Preview ${index + 1}`}
                                                    className="w-full h-full object-contain"
                                                />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <>
                                    <Upload className="w-6 h-6 text-gray-400 mx-auto mb-2" />
                                    <p className="text-sm text-gray-500 mb-2">
                                        Drop images here or click to select
                                    </p>
                                    <input
                                        type="file"
                                        multiple
                                        accept="image/*"
                                        onChange={handleFileSelect}
                                        className="hidden"
                                        id="file-upload"
                                    />
                                    <label
                                        htmlFor="file-upload"
                                        className="text-sm text-blue-600 hover:text-blue-700 cursor-pointer"
                                    >
                                        Select Files
                                    </label>
                                </>
                            )}
                        </div>

                        <FormField
                            control={form.control}
                            name="writerId"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Writer</FormLabel>
                                    <Select
                                        onValueChange={field.onChange}
                                        defaultValue={field.value}
                                    >
                                        <FormControl>
                                            <SelectTrigger>
                                                <SelectValue placeholder="Select a writer" />
                                            </SelectTrigger>
                                        </FormControl>
                                        <SelectContent>
                                            {writers?.map((writer) => (
                                                <SelectItem
                                                    key={writer.id}
                                                    value={writer.id.toString()}
                                                >
                                                    {writer.name}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </FormItem>
                            )}
                        />

                        <DialogFooter>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => onOpenChange(false)}
                                disabled={isUploading}
                            >
                                Cancel
                            </Button>

                            <Button
                                type="submit"
                                disabled={!form.watch('writerId') ||
                                    selectedFiles.length === 0 ||
                                    isUploading}
                            >
                                {isUploading ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"/>
                                        Processing...
                                    </>
                                ) : (
                                    'Upload Pages'
                                )}
                            </Button>
                        </DialogFooter>
                    </form>
                </Form>
            </DialogContent>
        </Dialog>
    );
}