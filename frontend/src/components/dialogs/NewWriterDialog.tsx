import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { writerApi } from '@/services/api';
import { useNavigate } from 'react-router-dom';
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { useForm } from "react-hook-form";

interface NewWriterDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

interface FormData {
    name: string;
    language: 'english' | 'german';
}

export default function NewWriterDialog({
                                            open,
                                            onOpenChange,
                                        }: NewWriterDialogProps) {
    const navigate = useNavigate();
    const queryClient = useQueryClient();

    const form = useForm<FormData>({
        defaultValues: {
            name: '',
            language: 'english',
        }
    });

    const createWriter = useMutation({
        mutationFn: async (data: {
            name: string;
            language: string;
        }) => {
            const response = await writerApi.create(data);
            return response.data;
        },
        onSuccess: (data) => {
            queryClient.invalidateQueries({ queryKey: ['writers'] });
            onOpenChange(false);
            form.reset();
            // Navigate to the new writer's detail page
            navigate(`/writers/${data.id}`);
        },
    });

    const handleSubmit = (data: FormData) => {
        createWriter.mutate({
            name: data.name,
            language: data.language,
        });
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>Create New Writer</DialogTitle>
                </DialogHeader>

                <Form {...form}>
                    <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
                        <FormField
                            control={form.control}
                            name="name"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Writer Name</FormLabel>
                                    <FormControl>
                                        <Input placeholder="Enter writer name..." {...field} />
                                    </FormControl>
                                </FormItem>
                            )}
                        />

                        <FormField
                            control={form.control}
                            name="language"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Primary Language</FormLabel>
                                    <Select
                                        onValueChange={field.onChange}
                                        defaultValue={field.value}
                                    >
                                        <FormControl>
                                            <SelectTrigger>
                                                <SelectValue placeholder="Select language" />
                                            </SelectTrigger>
                                        </FormControl>
                                        <SelectContent>
                                            <SelectItem value="english">English</SelectItem>
                                            <SelectItem value="german">German</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </FormItem>
                            )}
                        />

                        <div className="text-sm text-gray-500">
                            You can add training samples and start training after creating the writer.
                        </div>

                        <DialogFooter>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => onOpenChange(false)}
                                disabled={createWriter.isPending}
                            >
                                Cancel
                            </Button>
                            <Button
                                type="submit"
                                disabled={!form.watch('name').trim() || createWriter.isPending}
                            >
                                {createWriter.isPending ? 'Creating...' : 'Create Writer'}
                            </Button>
                        </DialogFooter>
                    </form>
                </Form>
            </DialogContent>
        </Dialog>
    );
}