import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
} from "@/components/ui/form";
import { Textarea } from "@/components/ui/textarea";
import { useForm } from "react-hook-form";
import { TrainingSample } from '@/services/api';
import { getStorageUrl } from '@/config/api';
import { useState, useEffect } from 'react';

interface EditTrainingSampleDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    sample: TrainingSample | null;
    onSubmit: (id: number, text: string) => void;
}

interface FormData {
    text: string;
}

export function EditTrainingSampleDialog({
                                             open,
                                             onOpenChange,
                                             sample,
                                             onSubmit
                                         }: EditTrainingSampleDialogProps) {
    const [showLines, setShowLines] = useState(false);
    const form = useForm<FormData>({
        defaultValues: {
            text: sample?.text || '',
        }
    });

    useEffect(() => {
        if (sample) {
            form.reset({
                text: sample.text
            });
        }
    }, [sample, form]);

    if (!sample) return null;

    // Handle line information if available
    const hasLineInfo = 'lines' in sample && Array.isArray(sample.lines) && sample.lines.length > 0;

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[600px]">
                <DialogHeader>
                    <DialogTitle>Edit Training Sample</DialogTitle>
                </DialogHeader>

                <Form {...form}>
                    <form onSubmit={form.handleSubmit((data) => onSubmit(sample.id, data.text))} className="space-y-4">
                        {/* Preview Image with Line Overlay */}
                        <div className="relative w-full h-48 bg-gray-100 rounded overflow-hidden">
                            <img
                                src={getStorageUrl(sample.image_path)}
                                alt="Training sample"
                                className="w-full h-full object-contain"
                            />
                            {hasLineInfo && showLines && (
                                <div className="absolute top-0 left-0 right-0 bottom-0">
                                    {(sample.lines || []).map((line, index) => (
                                        <div
                                            key={index}
                                            className="absolute border-2 border-blue-500 bg-blue-200/20"
                                            style={{
                                                left: `${line.bbox[0]}%`,
                                                top: `${line.bbox[1]}%`,
                                                width: `${line.bbox[2] - line.bbox[0]}%`,
                                                height: `${line.bbox[3] - line.bbox[1]}%`,
                                            }}
                                        />
                                    ))}
                                </div>
                            )}
                        </div>

                        {hasLineInfo && (
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => setShowLines(!showLines)}
                                className="w-full"
                            >
                                {showLines ? 'Hide Line Segments' : 'Show Line Segments'}
                            </Button>
                        )}

                        <FormField
                            control={form.control}
                            name="text"
                            render={({field}) => (
                                <FormItem>
                                    <FormLabel>Sample Text</FormLabel>
                                    <FormControl>
                                    <Textarea
                                            placeholder="Enter the text shown in the image..."
                                            className="min-h-[100px]"
                                            {...field}
                                        />
                                    </FormControl>
                                </FormItem>
                            )}
                        />

                        <DialogFooter>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => onOpenChange(false)}
                            >
                                Cancel
                            </Button>
                            <Button type="submit">
                                Save Changes
                            </Button>
                        </DialogFooter>
                    </form>
                </Form>
            </DialogContent>
        </Dialog>
    );
}