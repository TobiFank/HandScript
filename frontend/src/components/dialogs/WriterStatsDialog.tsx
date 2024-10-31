import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useQuery } from '@tanstack/react-query';
import { writerApi, type WriterStats } from '@/services/api';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Card, CardContent } from "@/components/ui/card";

interface WriterStatsDialogProps {
    writerId: number;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export default function WriterStatsDialog({
                                              writerId,
                                              open,
                                              onOpenChange,
                                          }: WriterStatsDialogProps) {
    const { data: stats } = useQuery<WriterStats>({
        queryKey: ['writer-stats', writerId],
        queryFn: async () => {
            const response = await writerApi.getStats(writerId);
            return response.data;
        },
    });

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-2xl">
                <DialogHeader>
                    <DialogTitle>Writer Statistics</DialogTitle>
                </DialogHeader>

                <div className="space-y-6">
                    {/* Accuracy Trend */}
                    <Card>
                        <CardContent className="pt-6">
                            <h3 className="text-sm font-medium mb-4">Recognition Accuracy Over Time</h3>
                            <div className="h-64">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={stats?.accuracy_trend || []}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="date" />
                                        <YAxis />
                                        <Tooltip />
                                        <Line
                                            type="monotone"
                                            dataKey="accuracy"
                                            stroke="#2563eb"
                                            strokeWidth={2}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Performance Metrics */}
                    <div className="grid grid-cols-2 gap-4">
                        <Card>
                            <CardContent className="pt-6">
                                <h4 className="text-sm font-medium text-muted-foreground">
                                    Average Processing Time
                                </h4>
                                <p className="text-2xl font-semibold mt-1">
                                    {stats?.avg_processing_time}ms
                                </p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardContent className="pt-6">
                                <h4 className="text-sm font-medium text-muted-foreground">
                                    Character Accuracy
                                </h4>
                                <p className="text-2xl font-semibold mt-1">
                                    {stats?.char_accuracy}%
                                </p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardContent className="pt-6">
                                <h4 className="text-sm font-medium text-muted-foreground">
                                    Word Accuracy
                                </h4>
                                <p className="text-2xl font-semibold mt-1">
                                    {stats?.word_accuracy}%
                                </p>
                            </CardContent>
                        </Card>
                        <Card>
                            <CardContent className="pt-6">
                                <h4 className="text-sm font-medium text-muted-foreground">
                                    Total Pages Processed
                                </h4>
                                <p className="text-2xl font-semibold mt-1">
                                    {stats?.total_pages}
                                </p>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}