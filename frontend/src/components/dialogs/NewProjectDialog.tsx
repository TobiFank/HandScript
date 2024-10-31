import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { projectApi } from '@/services/api';
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useForm } from "react-hook-form";

interface NewProjectDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

interface FormData {
    name: string;
    description: string;
}

export default function NewProjectDialog({
                                             open,
                                             onOpenChange,
                                         }: NewProjectDialogProps) {
    const queryClient = useQueryClient();
    const form = useForm<FormData>({
        defaultValues: {
            name: '',
            description: '',
        }
    });

    const createProject = useMutation({
        mutationFn: (data: { name: string; description?: string }) =>
            projectApi.create(data),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['projects'] });
            onOpenChange(false);
            form.reset();
        },
    });

    const handleSubmit = (data: FormData) => {
        createProject.mutate({
            name: data.name,
            description: data.description || undefined
        });
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Create New Project</DialogTitle>
                </DialogHeader>

                <Form {...form}>
                    <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-4">
                        <FormField
                            control={form.control}
                            name="name"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Project Name</FormLabel>
                                    <FormControl>
                                        <Input placeholder="Enter project name..." {...field} />
                                    </FormControl>
                                </FormItem>
                            )}
                        />

                        <FormField
                            control={form.control}
                            name="description"
                            render={({ field }) => (
                                <FormItem>
                                    <FormLabel>Description (Optional)</FormLabel>
                                    <FormControl>
                                        <Textarea
                                            placeholder="Enter description..."
                                            className="min-h-[80px]"
                                            {...field}
                                        />
                                    </FormControl>
                                </FormItem>
                            )}
                        />

                        {createProject.isError && (
                            <div className="text-red-600 text-sm">
                                Failed to create project. Please try again.
                            </div>
                        )}

                        <DialogFooter>
                            <Button
                                type="button"
                                variant="outline"
                                onClick={() => onOpenChange(false)}
                                disabled={createProject.isPending}
                            >
                                Cancel
                            </Button>
                            <Button
                                type="submit"
                                disabled={!form.watch('name').trim() || createProject.isPending}
                            >
                                {createProject.isPending ? 'Creating...' : 'Create Project'}
                            </Button>
                        </DialogFooter>
                    </form>
                </Form>
            </DialogContent>
        </Dialog>
    );
}